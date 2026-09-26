# walk_env.py — 讓機器人學走路的 Gymnasium 環境（1 台機器人，Isaac Sim 6.0.1）
# 注意：import 這個檔案之前，一定要先建立 SimulationApp（見 train.py / play.py）
import numpy as np
import gymnasium as gym
import omni.usd
from gymnasium import spaces
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

PHYS_DT, DECIM = 1 / 240, 4           # 物理 240 Hz，控制 60 Hz
KP, KD = 600.0, 20.0                  # 29 kg 的機器人（Track C 1.2 kg 用 20 / 0.4）
ACT_SCALE = 0.3                       # 動作 ±1 → 目標角 ±0.3 rad（再夾在關節限位內）
FWD = 1                               # 前方是 +Y（0925 版）；Track C 是 0（+X）
V_TARGET, EP_SECONDS = 0.15, 20.0     # 目標速度 (m/s)、每回合秒數
W = dict(alive=1.0, vel=2.0, upright=0.5, rate=0.02, fall=50.0)   # 獎勵權重（實作 5-3）


def rot_inv(q, v):
    """把世界座標的向量 v 轉到機身座標（q = w, x, y, z）"""
    w, u = q[0], q[1:]
    return v * (2 * w * w - 1) - np.cross(u, v) * 2 * w + u * 2 * np.dot(u, v)


class WalkEnv(gym.Env):
    def __init__(self, usd_path, render=False):
        self.render_on = render
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYS_DT,
                           rendering_dt=PHYS_DT * DECIM)
        mat = PhysicsMaterial(prim_path="/World/mat", static_friction=0.9,
                              dynamic_friction=0.8, restitution=0.0)
        GroundPlane(prim_path="/World/ground", size=200, physics_material=mat)
        stage = omni.usd.get_context().get_stage()
        if render:                                         # 開視窗才需要：燈、鏡頭
            UsdLux.DomeLight.Define(stage, "/World/light").CreateIntensityAttr(1000.0)
            set_camera_view(eye=[2.4, 1.0, 1.0], target=[0.0, 1.0, 0.3])

        prim = add_reference_to_stage(usd_path=usd_path, prim_path="/World/robot")
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        z_min = box.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2]
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0, 0, -z_min + 0.002))
        root = next(str(p.GetPath()) for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        self.robot = Articulation(prim_paths_expr=root, name="robot")
        self.world.scene.add(self.robot)
        self.world.reset()

        self.n = n = self.robot.num_dof
        self.robot.set_gains(kps=np.full((1, n), KP, np.float32),
                             kds=np.full((1, n), KD, np.float32))
        lim = np.asarray(self.robot.get_dof_limits())[0]             # (n, 2)，弧度
        self.lo, self.hi = lim[:, 0].astype(np.float32), lim[:, 1].astype(np.float32)
        self.q0 = np.zeros(n, np.float32)                  # 預設姿勢（改成微蹲會更好學）
        self.p0, self.r0 = self.robot.get_world_poses()    # 每回合的起點
        self.max_t = int(EP_SECONDS / (PHYS_DT * DECIM))
        obs_dim = 3 + 3 + n + n + n                        # 重力、角速度、角度、速度、上一個動作
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.set_world_poses(self.p0, self.r0)                  # 回到起點
        self.robot.set_velocities(np.zeros((1, 6), np.float32))       # 瞬移後速度要歸零
        self.robot.set_joint_positions(self.q0[None])
        self.robot.set_joint_velocities(np.zeros((1, self.n), np.float32))
        self.robot.set_joint_position_targets(self.q0[None])
        self.world.step(render=False)
        self.t, self.last_a = 0, np.zeros(self.n, np.float32)
        return self._obs(), {}

    def _obs(self):
        q = self.robot.get_world_poses()[1][0]                        # 機身姿態（w, x, y, z）
        self.g = rot_inv(q, np.array([0.0, 0.0, -1.0]))               # 重力方向（IMU）
        gyro = rot_inv(q, self.robot.get_angular_velocities()[0])    # 角速度（陀螺儀）
        qj = self.robot.get_joint_positions()[0] - self.q0            # 馬達角度
        qd = self.robot.get_joint_velocities()[0]                     # 馬達轉速
        obs = [self.g, 0.25 * gyro, qj, 0.05 * qd, self.last_a]
        return np.concatenate(obs).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        target = np.clip(self.q0 + ACT_SCALE * a, self.lo, self.hi)[None]
        for _ in range(DECIM):                             # 同一個目標，物理跑 4 步
            self.robot.set_joint_position_targets(target)
            self.world.step(render=False)
        if self.render_on:
            self.world.render()
        self.t += 1
        rate = float(np.sum((a - self.last_a) ** 2))
        self.last_a = a
        obs = self._obs()

        v = float(self.robot.get_linear_velocities()[0, FWD])  # 前進速度（真值，只算獎勵）
        upright = float(-self.g[2])                            # 1 = 完全直立
        fallen = upright < 0.707                               # 傾斜超過 45° 就算跌倒
        reward = (W["alive"]
                  + W["vel"] * np.exp(-((v - V_TARGET) / 0.1) ** 2)
                  + W["upright"] * upright
                  - W["rate"] * rate
                  - (W["fall"] if fallen else 0.0))
        truncated = self.t >= self.max_t
        return obs, float(reward), bool(fallen), bool(truncated), {}
