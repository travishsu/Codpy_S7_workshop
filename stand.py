# stand.py — 把機器人放到地上，用 PD 控制撐住站著
#   python stand.py                 讀 robot_usd.txt（import_urdf.py 寫的）
#   python stand.py xxx.usda        指定 USD
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

USD = sys.argv[1] if len(sys.argv) > 1 else open("robot_usd.txt").read().strip()
KP, KD = 600.0, 20.0                   # 29 kg：600 / 20（Track C 1.2 kg：20 / 0.4）

world = World(stage_units_in_meters=1.0, physics_dt=1 / 240, rendering_dt=1 / 60)
mat = PhysicsMaterial(prim_path="/World/mat", static_friction=0.9,
                      dynamic_friction=0.8, restitution=0.0)       # 地板不要彈
GroundPlane(prim_path="/World/ground", size=50, physics_material=mat)
stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/light").CreateIntensityAttr(1000.0)  # 沒燈會全黑
set_camera_view(eye=[1.2, 1.6, 0.8], target=[0.0, 0.0, 0.25])               # 鏡頭拉近

# 1. 放機器人：量出最低點，讓腳底離地 2 mm
prim = add_reference_to_stage(usd_path=USD, prim_path="/World/robot")
box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
z_min = box.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2]
UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0, 0, -z_min + 0.002))

# 2. 找到 articulation 的根，建立控制用的 view
root = next(str(p.GetPath()) for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI))
robot = Articulation(prim_paths_expr=root, name="robot")
world.scene.add(robot)
world.reset()

# 3. 設 PD 增益，目標 = URDF 的零度姿勢
n = robot.num_dof
robot.set_gains(kps=np.full((1, n), KP, np.float32),
                kds=np.full((1, n), KD, np.float32))
q0 = np.zeros((1, n), np.float32)
print("關節:", robot.dof_names, flush=True)

step = 0
while app.is_running():
    robot.set_joint_position_targets(q0)
    for _ in range(4):                  # 物理 240 Hz × 4 = 控制 60 Hz
        world.step(render=False)
    world.render()                      # 畫面更新一次（不推進物理）
    if step % 60 == 0:
        w, x, y, z = robot.get_world_poses()[1][0]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        print(f"t = {step / 60:5.1f} s   身體傾斜 = {tilt:4.1f}°", flush=True)
    step += 1
