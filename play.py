# play.py — 載入模型，開視窗看它走
#   python play.py checkpoints/walk_200000_steps.zip
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

from stable_baselines3 import PPO
from walk_env import WalkEnv

model = PPO.load(sys.argv[1] if len(sys.argv) > 1 else "walk_ppo", device="cpu")
env = WalkEnv(open("robot_usd.txt").read().strip(), render=True)
obs, _ = env.reset()
while app.is_running():
    action, _ = model.predict(obs, deterministic=False)   # 跟訓練時一樣帶雜訊
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
