import robotic as ry
import numpy as np
import sim_wrappers as sim
import matplotlib.pyplot as plt
import data_tools

class ManipNLP:
    view_speed = -1.

    def __init__(self, sim: sim.MjSim, target: sim.MjSimState, ctrl_pts: int):
        self.sim = sim
        self.x0 = sim.getState()
        self.q0 = sim.C.getJointState()
        self.qDim = sim.C.getJointDimension()
        self.ctrlPts = ctrl_pts
        self.dimension = self.qDim * self.ctrlPts
        self.featureTypes = [ry.OT.sos] * (self.x0.qpos.size) # + self.x0.qvel.size)
        self.bounds = np.array([np.full((self.dimension), -1.), np.full((self.dimension),+1.)])

        self.target = target

    def getInitializationSample(self):
        q = self.q0
        q = np.tile(q, (self.ctrlPts))
        return q

    def evaluate(self, x):
        # return (phi, J)

        splineX = x.reshape(-1, self.qDim)
        splineT = np.linspace(0, 1., splineX.shape[0]+1)
        splineT = splineT[1:]
        
        self.sim.setState(self.x0)
        self.sim.resetSplineRef(ctrl_time=0.)
        self.sim.setSplineRef(splineX, splineT, append=False)

        self.sim.step([], splineT[-1], ry.ControlMode.spline, view_speed=self.view_speed)

        x = self.sim.getState()
        self.sim.zeroQuatsFromQpos(x.qpos)
        self.sim.zeroQuatsFromQpos(self.target.qpos)
        phi1 = x.qpos - self.target.qpos
        # phi2 = x.qvel
        # phi = np.concatenate((phi1))
        phi = phi1

        J = np.zeros((0)) #undefined!
        return (phi, J)

def testOpt(rnd_poses, engine='mujoco'):
    tau_sim = .001
    
    C = ry.Config()
    C.addFile('sample/twoFingers.yml')
    if engine=='physx':
        S = ry.Simulation(C, engine=ry.SimulationEngine.physx, verbose=2)
    elif engine=='mujoco':
        S = sim.MjSim(open('sample/twoFingers.xml', 'r').read(), C, use_mj_viewer=False, tau_sim=tau_sim)
    else:
        raise Exception(f'engine "{engine}" not defined')

    tests = [(2,6), (2,1), (2,3)]
    test = tests[1]


    # set and get target state
    S.C.getFrame('obj').setPosition(rnd_poses[test[1],:3])
    S.C.setJointState(rnd_poses[test[1],3:])
    S.pushConfigToSim()
    x1 = S.getState()

    # set and get initial state
    S.C.getFrame('obj').setPosition(rnd_poses[test[0],:3])
    C.setJointState(rnd_poses[test[0],3:])
    S.pushConfigToSim()
    x0 = S.getState()

    nlp = ManipNLP(S, x1, 4)

    ry.params_add({
        'opt/stopEvals': 5000,
        'opt/stopTolerance': 1e-4,
        'LSZO/alpha_min': .01,
        'LSZO/maxIters': 2000,
        'LSZO/damping': 1e-2,
        'LSZO/noiseRatio': .2,
        'LSZO/noiseAbs': .000,
        # 'opt/stepDec': .7,
        # 'opt/stepInc': 1.,
        'opt/stepMax': .1,
        'LSZO/lambda': 1,
        'LSZO/mu': 1,
        'LSZO/dataRatio': 2.,
        'LSZO/pruneData': True,

        'LocalGreedy/sigma': .1,
        # 'LSZO/alpha_min': 1e-4,
        # 'LSZO/noiseRatio': .1,
        # 'LSZO/damping': 1e-1,
        # 'LSZO/covariantNoise': False
        # 'LSZO/dataRatio': 2.,
        # 'LSZO/pruneData': true,

        "CMA/sigmaInit": .01,
        "CMA/lambda": 10, #nlp.dimension,
        'LS_CMA/ls_lambda': 2,
        'GaussEDA/sigmaInit': .1,
        'GaussEDA/sigma2Min': 1e-10,
        'GaussEDA/beta': .2,
        'GaussEDA/momentum': .5,
        'ES/lambda': 10
        })

    sol = ry.NLP_Solver()
    sol.setPyProblem(nlp)
    # sol.setSolver(ry.OptMethod.LSZO)
    # sol.setSolver(ry.OptMethod.Rprop) .setOptions(finiteDifference=1e-4, verbose=2)
    # sol.setSolver(ry.OptMethod.Newton) .setOptions(finiteDifference=1e-4, verbose=2, stopTolerance=1e-6)
    # sol.setSolver(ry.OptMethod.LBFGS) .setOptions(finiteDifference=1e-4, verbose=2)
    # sol.setSolver(ry.OptMethod.NelderMead)
    # sol.setSolver(ry.OptMethod.CMA)
    sol.setSolver(ry.OptMethod.ES)
    ret = sol.solve(5)
    trace = sol.getTrace_costs()
    plt.plot(trace)
    plt.show()

    S.setState(x0)
    S.C.view(True, 'start')

    nlp.view_speed=1.
    phi, _ = nlp.evaluate(ret.x)
    S.C.view(True, 'end manip')
    x = S.getState()

    S.setState(x1)
    S.C.view(True, 'goal')

def main():
    np.random.seed()
    ry.rnd_seed_random()

    h5 = data_tools.H5Reader('rnd_twoFingers.h5')
    rnd_poses = h5.read('positions')
    print(type(rnd_poses), rnd_poses.shape)

    testOpt(rnd_poses)

if __name__ == "__main__":
    main()

