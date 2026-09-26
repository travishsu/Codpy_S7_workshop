# hello_isaac.py — 第一個 Isaac Sim 程式：一顆方塊掉到地上
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})        # 一定要最先建立

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()          # 預設地板 restitution = 0.8
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/cube", name="cube", size=0.2,
    position=np.array([0.0, 0.0, 1.0]), color=np.array([1.0, 0.0, 0.0])))
world.reset()

step = 0
while app.is_running():                         # 關掉視窗就結束
    world.step(render=True)
    if step % 30 == 0:
        z = cube.get_world_pose()[0][2]
        print(f"step {step:4d}   z = {z:.3f} m", flush=True)
    step += 1
app.close()
