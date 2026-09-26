# train.py — 用 PPO（Stable-Baselines3）訓練走路
#   python train.py                 讀 robot_usd.txt（import_urdf.py 寫的）
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})           # 訓練不開畫面，比較快

import carb                                        # 不把結果寫回 USD：快 2.5 倍
carb.settings.get_settings().set("/physics/updateToUsd", False)
carb.settings.get_settings().set("/physics/updateVelocitiesToUsd", False)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from walk_env import WalkEnv                       # 一定要在 SimulationApp 之後

usd = sys.argv[1] if len(sys.argv) > 1 else open("robot_usd.txt").read().strip()
env = WalkEnv(usd)
model = PPO("MlpPolicy", env,
            n_steps=2048, batch_size=256, n_epochs=5,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0),
            tensorboard_log="runs", device="cpu", verbose=1)
model.learn(total_timesteps=2_000_000,
            callback=CheckpointCallback(save_freq=50_000, save_path="checkpoints",
                                        name_prefix="walk"))
model.save("walk_ppo")
os._exit(0)
