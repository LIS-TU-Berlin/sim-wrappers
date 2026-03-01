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
    def __init__(self, t0, x0, v0, action_delta, lmbda, xi=1.):
        self.t0 = t0
        self.x0 = x0.copy()
        self.v0 = v0.copy()
        self.coeff  = (action_delta-2.*xi*lmbda*v0)/(2.*lmbda*lmbda) #see overleaf notes!

    def eval(self, t, single_th=-1):
        d = t - self.t0
        if single_th==-1:
            if isinstance(d, np.ndarray):
                return (d*d).reshape(-1,1)*self.coeff + d.reshape(-1,1)*self.v0 + self.x0
            else:
                return self.coeff*(d*d) + self.v0*d + self.x0
        else:
            return self.coeff[single_th]*(d*d) + self.v0[single_th]*d + self.x0[single_th]

    def eval_vel(self, t):
        d = t - self.t0
        return self.coeff*d + self.v0

    def reshape(self, num_threads):
        self.coeff = self.coeff.reshape(num_threads, -1)
        self.v0 = self.v0.reshape(num_threads, -1)
        self.x0 = self.x0.reshape(num_threads, -1)

    def reset(self, x0, th):
        assert self.x0.ndim==2
        self.coeff[th] *= 0.
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
        Basic simulation class that wraps a mujoco simulator.
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

        self.ctrlRef_poly = None
        self.ctrlRef_spline = None

        print(f"-- initialized MjSim with (controlled) joint dimension {C.getJointDimension()} and {len(self.freeobjs)} free objects (mj qpos:{self.data.qpos.size} qvel:{self.data.qvel.size} ctrl:{self.ctrl_dim})") # ctrl_indices:{self.ctrl_indices}
        
        assert self.data.qpos.size == self.C.getJointDimension()
        assert self.data.time == 0.

        self.pushConfigToSim()

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()

    def pushConfigToSim(self):
        """[internal] (re)set the mujoco state to be equal to the self.C state"""
        self.data.qpos = self.C.getJointState()

        mujoco.mj_forward(self.model, self.data)

        if self.use_mj_viewer:
            self.viewer.sync()

    def pullConfigFromSim(self):
        """[interna] set selt.C state equal to mujoco state"""
        self.C.setJointState(self.data.qpos)

    def get_ctrlRef(self):
        if self.ctrlRef_poly is not None:
            cref = self.ctrlRef_poly.eval(self.ctrl_time)
        elif self.ctrlRef_spline is not None:
            cref = self.ctrlRef_spline.eval3(self.ctrl_time)[0]
        else:
            raise Exception('you need to set a ctrl reference')
        return cref

    def multi_sim_steps(self, steps: int) -> None:
        view_steps = math.ceil(0.02 / self.tau_sim * self.view_speed)
        for k in range(steps):
            cref = self.get_ctrlRef()
            self.data.ctrl = cref.reshape(-1)

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

        mujoco.mj_forward(self.model, self.data)
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

    def resetPolyRef(self, ctrl_time: float = 0.) -> None:
        cref = self.data.qpos[self.ctrl_indices]
        self.ctrlRef_spline = None
        self.ctrlRef_poly = SecondOrderCtrlRef(ctrl_time, cref, np.zeros(cref.shape), np.zeros(cref.shape), 1.)
        self.ctrl_time = ctrl_time

    def resetSplineRef(self, ctrl_time: float = 0., const_ref=None) -> None:
        """[core] reset the spline; ctrl_time gives the *absolute* time (relating to mujoco's time state) of the spline knots"""
        self.ctrlRef_poly = None
        self.ctrlRef_spline = ry.BSpline()
        if const_ref is None:
            cref = self.data.qpos[self.ctrl_indices]
        else:
            cref = const_ref
        self.ctrlRef_spline.set(2, cref.reshape(1, -1), [ctrl_time])
        self.ctrl_time = ctrl_time

    def updateSplineRef(self, points: np.array, times: np.array, append: bool = False) -> None:
        """[core] set the spline; when overwriting, times are relative to the *current* ctrl_time"""
        self.ctrlRef_poly = None
        if not append:
            self.ctrlRef_spline.overwriteSmooth(points, times, self.ctrl_time)
        else:
            raise NotImplementedError()

    def updatePolyRef(self, delta, time_horizon):
        raise NotImplementedError()
        current_vel = self.data.qvel[: self.ctrl_dim]
        current_ref = self.ctrlRef_poly.eval(self.ctrl_time)
        self.ctrlRef_poly = SecondOrderCtrlRef(self.ctrl_time, current_ref, current_vel, delta, time_horizon)

    def get_Jacobian(self, frame_name):
        body_id = mujoco.mj_name2id(self.model, 1, frame_name)
        assert body_id>=0, f'frame name {frame_name} is not a mj body'
        pos = self.data.xpos[body_id]
        Jpos = np.empty((3, self.model.nv))
        Jang = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, Jpos, Jang, pos, body_id)
    
        if False: #test
            self.pullConfigFromSim()
            y, J = self.C.eval(ry.FS.position, [frame_name])
            print(np.linalg.norm(pos-y))
            print(Jpos, '\n', J)

        return Jpos, Jang

    def get_position(self, frame_name):
        body_id = mujoco.mj_name2id(self.model, 1, frame_name)
        assert body_id>=0, f'frame name {frame_name} is not a mj body'
        pos = self.data.xpos[body_id]
        return pos

    @property
    def qpos_dim(self) -> int:
        return self.data.qpos.size
    
    @property
    def qvel_dim(self) -> int:
        return self.data.qvel.size
    
    @property
    def ctrl_dim(self) -> int:
        return self.data.ctrl.size
