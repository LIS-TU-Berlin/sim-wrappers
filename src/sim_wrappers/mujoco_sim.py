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

class MujocoSim:
    mj_steps = 0
    ctrl_time = 0.
    ctrl_costs = 0.
    qpos_offset = None #an offset before set/getState

    # for inspection & rendering only
    view_speed = -1.
    save_qpos = -1
    saved_qpos = []
    save_images = False
    saved_images = []

    def __init__(
        self,
        xml_description: str,
        C: ry.Config,
        use_mj_viewer: bool = True,
        tau_sim: float = 1e-3
    ):
        """
        Basic simulation class that wraps a mujoco simulator.
        """
        self.model = mujoco.MjModel.from_xml_string(xml_description)
        self.data = mujoco.MjData(self.model)
        self.tau_sim = tau_sim
        self.model.opt.timestep = self.tau_sim
        self.use_mj_viewer = use_mj_viewer

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

        self.ctrl_indices = []
        for i in range(self.ctrl_dim):
            id = self.model.actuator(i).trnid[0]
            qid = self.model.joint(id).qposadr
            self.ctrl_indices.append(int(qid[0]))
        self.ctrl_indices = np.array(self.ctrl_indices, dtype='int32')

        self.ctrl_buffer = None
        self.ctrl_bufferPtr = 0
        self.ctrlRef_spline = None

        print(f"-- initialized MjSim with (controlled) joint dimension {C.getJointDimension()} (mj qpos:{self.data.qpos.size} qvel:{self.data.qvel.size} ctrl:{self.ctrl_dim})") # ctrl_indices:{self.ctrl_indices}
        
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
        if self.ctrlRef_spline is not None:
            cref = self.ctrlRef_spline.eval3(self.ctrl_time)[0]
        else:
            raise Exception('you need to set a ctrl reference')
        return cref

    def multi_step(self, num_steps: int) -> None:
        view_steps = math.ceil(0.02 / self.tau_sim * self.view_speed)
        for k in range(num_steps):
            assert self.ctrl_buffer is not None
            assert self.ctrl_buffer.shape[0]>self.ctrl_bufferPtr , 'ctrlRef buffer too small'
            cref = self.ctrl_buffer[self.ctrl_bufferPtr]
            self.ctrl_bufferPtr += 1

            # cref = self.get_ctrlRef()
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
        self.multi_step(sim_steps)

    def getState(self) -> MjSimState:
        """[core] get a state struct that allows exact reset"""
        return MjSimState(
            time=self.data.time,
            qpos=self.qpos_minus_offset.copy(),
            qvel=self.data.qvel.copy(),
            act=self.data.actuator_force.copy(),
        )

    def setState(self, state: MjSimState) -> None:
        """[core] set the state"""
        self.data.time = state.time
        if self.qpos_offset is None:
            self.data.qpos[:] = state.qpos
        else:
            self.data.qpos[:] = state.qpos + self.qpos_offset
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
        self.ctrlRef_spline = ry.BSpline()
        if const_ref is None:
            cref = self.data.qpos[self.ctrl_indices]
        else:
            cref = const_ref
        self.ctrlRef_spline.set(2, cref.reshape(1, -1), [ctrl_time])
        self.ctrl_time = ctrl_time

    def updateSplineRef(self, points: np.array, times: np.array, append: bool = False) -> None:
        """[core] set the spline; when overwriting, times are relative to the *current* ctrl_time"""
        if not append:
            self.ctrlRef_spline.overwriteSmooth(points, times, self.ctrl_time)
        else:
            raise NotImplementedError()

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

    def get_free_objects(self):
        freeobjs = []
        for f in self.C.getFrames():
            if f.getParent() == None or (f.getJointType() == ry.JT.free):
                # print(f.name)
                if "mass" in f.asDict():
                    freeobjs.append(f)
        return freeobjs

    @property
    def qpos_minus_offset(self) -> np.array:
        if self.qpos_offset is not None:
            return self.data.qpos-self.qpos_offset
        else:
            return self.data.qpos
        
    @property
    def qpos_dim(self) -> int:
        return self.data.qpos.size
    
    @property
    def qvel_dim(self) -> int:
        return self.data.qvel.size
    
    @property
    def ctrl_dim(self) -> int:
        return self.data.ctrl.size
