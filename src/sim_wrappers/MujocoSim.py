# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import mujoco.viewer
import numpy as np
import time, math
import robotic as ry

class MjSimState:
    def __init__(self, mj_data: mujoco.MjData):
        self.time = mj_data.time
        self.qpos = mj_data.qpos.copy()
        self.qvel = mj_data.qvel.copy()
        self.act = mj_data.act.copy()

class MjSim:
    LQR_K = None
    LQR_const = None

    def __init__(self, xml, C: ry.Config, use_mj_viewer=True, tau_sim=0.01):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = tau_sim
        self.tau_sim = tau_sim

        if use_mj_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.freeobjs = []
        for f in C.getRoots():
            if 'mass' in f.asDict():
                self.freeobjs.append(f)
        print(f'-- initializing MjSim with {len(self.freeobjs)} free objects and joint dimension {C.getJointDimension()} (mj qpos:{self.data.qpos.size} ctrl:{self.data.ctrl.size})')
        self.C = C
        self.pushConfigToSim()

        assert self.data.qpos.size == self.C.getJointDimension() + 7 * len(self.freeobjs)
        assert self.data.ctrl.size == self.C.getJointDimension()

        self.ctrl_dim = self.data.ctrl.size
        self.resetSplineRef(0.)

    def __del__(self):
        if self.viewer is not None:
            self.viewer.close()

    def pushConfigToSim(self):
        """[internal] (re)set the mujoco state to be equal to the self.C state"""
        qn = self.C.getJointDimension()
        self.data.qpos[:qn] = self.C.getJointState()
        self.data.qvel[:qn] = np.zeros(qn)

        for i, f in enumerate(self.freeobjs):
            self.data.qpos[qn + 7 * i : qn + 7 * (i + 1)] = f.getPose()
            self.data.qvel[qn + 6 * i : qn + 6 * (i + 1)] = np.zeros(6)

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        # mujoco.mj_setState(self.m,self.d,q, mujoco.mjtState.mjSTATE_QPOS)
        # mujoco.mj_setState(m,d,q, mujoco.mjtState.mjSTATE_CTRL)

    def pullConfigFromSim(self):
        """[interna] set selt.C state equal to mujoco state"""
        qn = self.C.getJointDimension()

        self.C.setJointState(self.data.qpos[:qn])

        for i, f in enumerate(self.freeobjs):
            f.setPose(self.data.qpos[qn + 7 * i : qn + 7 * (i + 1)])

    def multi_steps(self, steps, view_speed=-1.):
        viewSteps = math.ceil(0.03 / self.tau_sim / view_speed)
        for k in range(steps):
            if self.LQR_K is not None:
                self.data.ctrl = self.data.qpos[:self.ctrl_dim]
                feedback = self.LQR_K @ np.concatenate((self.getQPosWithoutQuatW(), self.data.qvel)) + self.LQR_const 
                self.data.ctrl += 1.*feedback
            else:
                ref = self.ctrl.eval3(self.ctrl_time)
                self.data.ctrl = ref[0]
            mujoco.mj_step(self.model, self.data)
            self.ctrl_time += self.tau_sim
            if view_speed>0. and ((k+1)%viewSteps==0 or k==steps-1):
                if self.viewer is not None:
                    self.viewer.sync()
                self.pullConfigFromSim()
                self.C.view(False, f"mujoco sim time: {self.data.time:6.3f}, ctrl time: {self.ctrl_time:6.3f}")
                time.sleep(view_speed * viewSteps * self.tau_sim)
        self.pullConfigFromSim()

    def step(self, u, tau_step, mode=None, view_speed=-1.):
        """[core] step the physics engine"""
        steps = round(tau_step/self.tau_sim)
        assert math.isclose(tau_step, steps*self.tau_sim), 'tau_step needs to be a multiple of tau_sim'
        self.multi_steps(steps, view_speed)

    def getState(self):
        """[core] get a state struct that allows exact reset"""
        return MjSimState(self.data)

    def setState(self, state: MjSimState):
        """[core] set the state"""
        self.data.time = state.time
        self.data.qpos = state.qpos
        self.data.qvel = state.qvel
        self.data.act = state.act
        mujoco.mj_forward(self.model, self.data)
        self.ctrl_time = state.time
        if self.viewer is not None:
            self.viewer.sync()
        self.pullConfigFromSim()

    def resetSplineRef(self, ctrl_time):
        """[core] reset the spline; ctrl_time gives the *absolute* time (relating to mujoco's time state) of the spline knots"""
        self.ctrl = ry.BSpline()
        ref = self.data.qpos
        ref = ref[:self.data.ctrl.size]
        self.ctrl.set(2, ref.reshape(1, -1), [ctrl_time])
        self.ctrl_dim = self.data.ctrl.size
        self.ctrl_time = ctrl_time

    def setSplineRef(self, points, times, append: bool):
        """[core] set the spline; when overwriting, times are relative to the *current* ctrl_time"""
        if not append:
            self.ctrl.overwriteSmooth(points, times, self.ctrl_time)
        else:
            raise NotImplementedError()
        
    def getLinearizedSystem(self, throughPD=False, kp=None, kd=None):
        """[to be moved]"""
        ref = self.ctrl.eval3(self.ctrl_time)
        self.data.ctrl = ref[0]
        mujoco.mj_forward(self.model, self.data)
        nv = self.data.qvel.size
        nu = self.ctrl_dim
        F = np.empty((nv,1))
        mujoco.mj_rne(self.model, self.data, 0, F)
        A = np.empty((2*nv,2*nv))
        B = np.empty((2*nv,nu))
        mujoco.mjd_transitionFD(self.model, self.data, eps=1e-4, flg_centered=0, A=A, B=B, C=None, D=None)
        A -= np.eye(A.shape[0])
        A /= self.tau_sim
        B /= self.tau_sim
        
        return A, B, F.reshape(-1)

    def getQPosWithoutQuatW(self):
        """[to be moved]"""
        qn = self.C.getJointDimension()
        nobj = len(self.freeobjs)
        qpos = np.empty((qn + 6*nobj))
        qpos[:qn] = self.data.qpos[:qn]
        for i in range(nobj):
            qpos[qn+6*i+0 : qn+6*i+3] = self.data.qpos[qn+7*i+0 : qn+7*i+3]
            qpos[qn+6*i+3 : qn+6*i+6] = self.data.qpos[qn+7*i+3 : qn+7*i+6]
        return qpos
    
    def zeroQuatsFromQpos(self, qpos):
        """[to be moved]"""
        qn = self.C.getJointDimension()
        for i in range(len(self.freeobjs)):
            qpos[qn+7*i+4 : qn+7*i+7] = 0
        return qpos

