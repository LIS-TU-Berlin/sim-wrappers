import sim_wrappers as sim
import robotic as ry
import numpy as np

def testActionReset(engine='mujoco'):
    tau_sim = .001
    tau_action = .1
    tau_spline = .2

    C = ry.Config()
    C.addFile('sample/twoFingers.yml')
    if engine=='physx':
        S = ry.Simulation(C, engine=ry.SimulationEngine.physx, verbose=2)
    elif engine=='mujoco':
        S = sim.MjSim(open('sample/twoFingers.xml', 'r').read(), C, view=False, tau_sim=tau_sim)
    else:
        raise Exception(f'engine "{engine}" not defined')

    for _ in range(10):

        # set initial state
        x0 = S.getState()

        # random target and spline ref
        q0 = S.C.getJointState()
        q_target = q0 + .1 * np.random.randn(q0.size)

        # simulate a step
        S.resetSplineRef(ctrl_time=0.)
        S.setSplineRef(q_target.reshape(1,-1), [tau_spline], append=False)
        S.step([], tau_step=tau_action, mode=ry.ControlMode.spline, view=1.)

        # store result
        q1 = S.C.getJointState()
        print('q1: ', q1)
        S.C.view(True, 'original action')

        for _ in range(2):
            #reset & step same action
            S.setState(x0)
            S.resetSplineRef(ctrl_time=0.)
            S.setSplineRef(q_target.reshape(1,-1), [tau_spline], append=False)
            S.step([], tau_step=tau_action, mode=ry.ControlMode.spline, view=1.)

            # compare result
            q2 = S.C.getJointState()
            print('q2: ', q2, 'error: ', np.linalg.norm(q1-q2))
            S.C.view(True, 'repeated action (after state reset)')

def main():
    np.random.seed()
    ry.rnd_seed_random()

    testActionReset()

if __name__ == "__main__":
    main()

