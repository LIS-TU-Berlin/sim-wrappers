# https://mujoco.readthedocs.io/en/stable/python.html
from typing import Optional
from dataclasses import dataclass

#from chex import Array
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import robotic as ry


@dataclass
class MjSimState:
    time: float
    qpos: np.array
    qvel: np.array
    act: np.array
    
    def as_vector(self):
        return np.concat((np.array([self.time]), self.qpos, self.qvel, self.act))

class SecondOrderCtrlRef:
    def __init__(self, t0, x0, v0, delta, tau):
        self.t0 = t0
        self.x0 = x0.copy()
        self.v0 = v0.copy()
        self.a  = (delta-tau*v0)/(tau*tau) #at time t=tau, f(t) = x0+delta

    def eval(self, t, single_th=-1):
        d = t - self.t0
        if single_th==-1:
            return self.a*(d*d) + self.v0*d + self.x0
        else:
            return self.a[single_th]*(d*d) + self.v0[single_th]*d + self.x0[single_th]
    
    def reshape(self, num_threads):
        self.a = self.a.reshape(num_threads, -1)
        self.v0 = self.v0.reshape(num_threads, -1)
        self.x0 = self.x0.reshape(num_threads, -1)

    def reset(self, x0, th):
        assert self.x0.ndim==2
        self.a[th] *= 0.
        self.v0[th] *= 0.
        self.x0[th] = x0

class MujocoSim:
    mj_steps = 0
    ctrl_time = 0.
    ctrl_costs = 0.

    view_speed = -1.
    save_qpos = -1
    saved_qpos = []
    save_images = False
    saved_images = []

    def __init__(
        self,
        xml_path: str,
        C: ry.Config,
        use_mj_viewer: bool = True,
        tau_sim: float = 1e-3
    ):
        """
        Basic simulation class that wraps a mujoco simulator and rai config.

        :param xml_path: String or pathlike object that points to the xml file.
        :param C: rai configuration of the scene.
        :param use_mj_viewer: Use the mujoco native viewer.
        :param tau_sim: Simulation step. Will be set so model.opt.timestep in mujoco equals this
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.tau_sim = tau_sim
        self.model.opt.timestep = self.tau_sim
        self.use_mj_viewer = use_mj_viewer
        self.xml_file = xml_path

        if use_mj_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            cam_pose = C.get_viewer().getCamera_pose()
            q = ry.Quaternion()
            q.set(cam_pose[3:])
            rpy = q.getRollPitchYaw()
            self.viewer.cam.azimuth = -90.+rpy[1]*180./math.pi
            self.viewer.cam.elevation = 90.-rpy[0]*180./math.pi
            self.viewer.cam.distance = 3.
            self.viewer.cam.lookat = [0., 0., .5]
        else:
            self.viewer = None

        self.C = C
        self.freeobjs = []
        for f in C.getFrames():
            if f.getParent() == None or (f.getJointType() == ry.JT.free):
                # print(f.name)
                if "mass" in f.asDict():
                    self.freeobjs.append(f)

        self.ctrl_indices = []
        for i in range(self.ctrl_dim):
            id = self.model.actuator(i).trnid[0]
            qid = self.model.joint(id).qposadr
            self.ctrl_indices.append(int(qid[0]))
        self.ctrl_indices = np.array(self.ctrl_indices, dtype='int32')

        assert self.data.qpos.size == self.C.getJointDimension() #+ 7 * len(self.freeobjs)
        # assert self.ctrl_dim == self.C.getJointDimension()
        assert self.data.time == 0.

        self.pushConfigToSim()
        # self.resetSplineRef(ctrl_time=0.)

        print(f"-- initialized MjSim with (controlled) joint dimension {C.getJointDimension()} and {len(self.freeobjs)} free objects (mj qpos:{self.data.qpos.size} qvel:{self.data.qvel.size} ctrl:{self.ctrl_dim})") # ctrl_indices:{self.ctrl_indices}

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()

    def pushConfigToSim(self):
        """[internal] (re)set the mujoco state to be equal to the self.C state"""
        self.data.qpos = self.C.getJointState()
        # self.data.qpos[: self.q_dim] = self.C.getJointState()
        # self.data.qvel[: self.q_dim] = np.zeros(self.q_dim)
        # for i, f in enumerate(self.freeobjs):
        #     self.data.qpos[self.q_dim + 7 * i : self.q_dim + 7 * (i + 1)] = f.getPose()
        #     self.data.qvel[self.q_dim + 6 * i : self.q_dim + 6 * (i + 1)] = np.zeros(6)

        mujoco.mj_forward(self.model, self.data)

        if self.use_mj_viewer:
            self.viewer.sync()

        # mujoco.mj_setState(self.m,self.d,q, mujoco.mjtState.mjSTATE_QPOS)
        # mujoco.mj_setState(m,d,q, mujoco.mjtState.mjSTATE_CTRL)

    def pullConfigFromSim(self):
        """[interna] set selt.C state equal to mujoco state"""
        self.C.setJointState(self.data.qpos)
        # self.C.setJointState(self.data.qpos[: self.q_dim])
        # for i, f in enumerate(self.freeobjs):
        #     f.setPose(self.data.qpos[self.q_dim + 7 * i : self.q_dim + 7 * (i + 1)])

    def multi_sim_steps(self, steps: int) -> None:
        view_steps = math.ceil(0.02 / self.tau_sim * self.view_speed)
        for k in range(steps):
            # ctrl_ref = self.spline_ref.eval3(self.ctrl_time)[0]
            ctrl_ref = self.ctrlRef.eval(self.ctrl_time)
            self.data.ctrl = ctrl_ref.reshape(-1)

            mujoco.mj_step(self.model, self.data)
            self.mj_steps += 1
            self.ctrl_time += self.tau_sim
            self.ctrl_costs += np.sum(np.square(self.data.actuator_force))

            # storing the path
            if self.save_qpos>0 and (self.mj_steps%self.save_qpos==0):
                self.saved_qpos.append(self.data.qpos.copy())

            # visualization, and storing images
            if self.view_speed > 0.0 and (self.mj_steps%view_steps==0):
                if self.use_mj_viewer:
                    self.viewer.sync()
                self.pullConfigFromSim()
                self.C.view(False, f"sim t:{self.ctrl_time:6.3f}", offscreen=self.save_images)
                if self.save_images:
                    self.saved_images.append(self.C.get_viewer().getRgb())
                time.sleep(view_steps * self.tau_sim / self.view_speed)
        self.pullConfigFromSim()

    def step(self, tau_step: float) -> None:
        """[core] step the physics engine for a given time, usually making multiple small (tau_sim) steps"""
        sim_steps = round(tau_step / self.tau_sim)
        assert math.isclose(tau_step, sim_steps * self.tau_sim), "tau_step needs to be a multiple of tau_sim"
        self.multi_sim_steps(sim_steps)

    def getState(self) -> MjSimState:
        """[core] get a state struct that allows exact reset"""
        return MjSimState(
            time=self.data.time,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            act=self.data.actuator_force.copy(),
        )

    def setState(self, state: MjSimState) -> None:
        """[core] set the state"""
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        self.data.actuator_force[:] = state.act
        mujoco.mj_forward(self.model, self.data)
        self.ctrl_time = state.time
        if self.use_mj_viewer:
            self.viewer.sync()
        self.pullConfigFromSim()

    def to_state(self, s: np.array) -> MjSimState:
        nq, nv = self.data.qpos.size, self.data.qvel.size
        assert s.size==1+nq+nv+self.ctrl_dim, "wrong size"
        return MjSimState(s[0], s[1:1+nq], s[1+nq:1+nq+nv], s[1+nq+nv:])

    def resetSplineRef(self, ctrl_time: float = 0., const_ref=None) -> None:
        """[core] reset the spline; ctrl_time gives the *absolute* time (relating to mujoco's time state) of the spline knots"""
        # self.spline_ref = ry.BSpline()
        raise NotImplementedError()
        if const_ref is None:
            ref = self.data.qpos[self.ctrl_indices]
        else:
            assert const_ref.size==self.ctrl_dim
            ref = const_ref
        # self.spline_ref.set(2, ref.reshape(1, -1), [ctrl_time])
        n = ref.shape[0]
        self.ctrlRef = SecondOrderCtrlRef(ctrl_time, ref, np.zeros((n)), np.zeros((n)), 1.)
        self.ctrl_time = ctrl_time

    def setSplineRef(self, points: np.array, times: np.array, append: bool = False) -> None:
        """[core] set the spline; when overwriting, times are relative to the *current* ctrl_time"""
        # if not append:
        #     self.spline_ref.overwriteSmooth(points, times, self.ctrl_time)
        # else:
        # NEW:
        raise NotImplementedError()

    def updateCtrlRef(self, delta, time_horizon):
        raise NotImplementedError()
        current_vel = self.data.qvel[: self.ctrl_dim]
        current_ref = self.ctrlRef.eval(self.ctrl_time)
        self.ctrlRef = SecondOrderCtrlRef(self.ctrl_time, current_ref, current_vel, delta, time_horizon)

    def getLinearizedSystem(
        self,
        through_pd: bool = False,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> tuple[np.array, np.array, np.array]:
        """[to be moved]"""
        ref = self.spline_ref.eval3(self.ctrl_time)
        self.data.ctrl = ref[0]
        mujoco.mj_forward(self.model, self.data)
        nv = self.data.qvel.size
        nu = self.ctrl_dim
        F = np.empty((nv, 1))
        mujoco.mj_rne(self.model, self.data, 0, F)
        A = np.empty((2 * nv, 2 * nv))
        B = np.empty((2 * nv, nu))
        mujoco.mjd_transitionFD(
            self.model, self.data, eps=1e-4, flg_centered=0, A=A, B=B, C=None, D=None
        )
        A -= np.eye(A.shape[0])
        A /= self.tau_sim
        B /= self.tau_sim

        return A, B, F.reshape(-1)

    def getQPosWithoutQuatW(self) -> np.array:
        """[to be moved]"""
        qn = self.C.getJointDimension()
        nobj = len(self.freeobjs)
        qpos = np.empty((qn + 6 * nobj))
        qpos[:qn] = self.data.qpos[:qn]
        for i in range(nobj):
            qpos[qn + 6 * i + 0 : qn + 6 * i + 3] = self.data.qpos[
                qn + 7 * i + 0 : qn + 7 * i + 3
            ]
            qpos[qn + 6 * i + 3 : qn + 6 * i + 6] = self.data.qpos[
                qn + 7 * i + 3 : qn + 7 * i + 6
            ]
        return qpos

    def zeroQuatsFromQpos(self, qpos: np.array) -> np.array:
        """[to be moved]"""
        qn = self.C.getJointDimension()
        for i in range(len(self.freeobjs)):
            qpos[qn + 7 * i + 3 : qn + 7 * i + 7] = 0
        return qpos

    @property
    def qpos_dim(self) -> int:
        return self.data.qpos.size
    
    @property
    def qvel_dim(self) -> int:
        return self.data.qvel.size
    
    @property
    def ctrl_dim(self) -> int:
        return self.data.ctrl.size

    @property
    def q_dim(self) -> int:
        return self.C.getJointDimension()
