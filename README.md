# Codpy Section 7 · Isaac 體驗 — 範例程式

簡報裡每一張程式碼投影片，都是從這個資料夾的檔案原文擷取的（有幾張省略了整行註解與空行）。
預設對象是學員用的 **Biped_Robot_Master 0925 版**（Drive「雙足機器人_髖關節連接件修正」）。

## 環境

| 套件 | 版本 |
|---|---|
| Isaac Sim | 6.0.1（`pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com`）——pypi 上最新已是 6.1.0.0，**版本一定要寫** |
| Python | 3.12 |
| PyTorch | 2.11.0 + cu128（`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`） |
| 訓練 | `pip install stable-baselines3 tensorboard`（講者環境：stable-baselines3 2.9.0、gymnasium 1.3.0、tensorboard 2.21.0） |

## 準備機器人

1. 打開 go.ncku.net/1ZY8Hd →「雙足機器人_髖關節連接件修正」。
2. 下載整個 `Biped_Robot_Master_0925_unity_urdf` 資料夾（裡面有 `.urdf` 和 `Biped_Robot_Master_0925_description/meshes/`，網格是相對路徑，不能只拿 `.urdf`）。
3. 放進這個資料夾。**整條路徑只用英文、不要有空格**（例如 `C:\workshop`）。

## 使用順序

終端機在這個資料夾執行。`import_urdf.py` 會把 USD 的路徑寫進 `robot_usd.txt`，後面的程式預設都讀它。

| 時段 | 指令 | 做什麼 |
|---|---|---|
| Part 2 | `python hello_isaac.py` | 開視窗，一顆紅色方塊掉到藍色格線地板上（預設地板會彈；第一次要連網下載地板） |
| Part 3 | `python import_urdf.py` | 自動找到資料夾裡唯一的 `.urdf` → `usd/<名字>/<名字>.usda`，寫 `robot_usd.txt` |
| Part 3 | `python stand.py` | 放到 restitution 0 的地板上（有燈、鏡頭拉近），用 PD 撐住站著，印出身體傾斜角 |
| Part 5 | `python train.py` | PPO 訓練，`runs/` 給 TensorBoard，`checkpoints/` 每 5 萬步存一次；停止按 Ctrl+C（沒反應就關掉終端機） |
| Part 5 | `tensorboard --logdir runs` | 另開終端機，瀏覽器開 `localhost:6006`，看 `rollout/ep_rew_mean`、`rollout/ep_len_mean` |
| Part 5 | `python play.py checkpoints/walk_300000_steps.zip` | 開視窗播放某個 checkpoint（跟訓練時一樣帶雜訊） |

`walk_env.py` 是訓練環境（被 `train.py` / `play.py` import），實作 5-3 要改的獎勵權重是檔案上方的 `W`。

## 0925 版量到的數字，以及它們在程式裡的位置

直接從 0925 的 URDF 和 STL 算的（零度姿勢），並在 Isaac Sim 6.0.1 裡確認過：

| 量到的 | 數值 | 程式裡 |
|---|---|---|
| 總重 / 單腳 | 29.05 kg / 6.03 kg，重心高 0.19 m | `KP, KD = 600.0, 20.0`（class_sim 對這台用的增益） |
| 前方 | 腳尖往 +Y 伸 125 mm、腳跟往 −Y 55 mm → **+Y** | `walk_env.py` 的 `FWD = 1` |
| base_link 原點 | 比腳底低 4.8 mm（匯入後實測根的高度 −0.0048 m） | 跌倒只看傾斜角（`upright < 0.707`），不用高度 |
| 關節 | 10 個會轉（每腿 4 + 腰 2）、2 個固定；髖 roll 與踝 ±20° | 目標角夾在 `get_dof_limits()` 內；observation 36 維、action 10 維 |
| 關節改名 | `Rigid 41` → `tn__Rigid41_i7`（名字有空格的才會被改） | 用 (parent, child) 或 `dof_names` 找關節 |
| 自碰撞 | 零度姿勢時，不相鄰零件的凸包沒有重疊；兩腳內緣相距 140 mm | PhysX 預設就會自碰撞；`allow_self_collision` 在 6.0.1 只寫給 Newton |
| 碰撞形狀 | 全部是 STL 網格 | `collision_type="Convex Hull"`（匯入後 13 個 convexHull） |
| 能不能站 | 重心在兩腳支撐範圍內 69.5 mm | `stand.py` 目標 = 全部 0° |

注意：0925 的 URDF 左右小腿不對稱（`J_Left_Shin_Link` 的 origin x 是 −0.022，右邊是 0.0），左踝比右踝多往外 20 mm。

換成 Track C（`humanoid_c.urdf`）要改：`KP, KD = 20.0, 0.4`、`FWD = 0`，並在 PhysX 的 articulation 上把
`enabledSelfCollisions` 設成 False（`isaac-sim-c/sim_c.py` 的做法；匯入器的 `allow_self_collision=False` 對 PhysX 沒有作用）。

## 驗證（2026-09-27，講者筆電 RTX 3060 Laptop 6 GB、RAM 16 GB，Isaac Sim 6.0.1）

照簡報的步驟、用這個資料夾裡**原封不動的檔案**跑過一遍：

| 步驟 | 結果 |
|---|---|
| `hello_isaac.py` | 視窗顯示紅色方塊落在格線地板上；方塊彈跳約 14 cm → 2 cm → 0.5 cm 後停在 z = 0.100 m；啟動約 11 秒 |
| `import_urdf.py` | 21 秒完成；USD 的質量 29.049 kg（13/13 個 link）、10 個關節限位都和 URDF 一致 |
| `stand.py` | 視窗看得到機器人（有燈、鏡頭拉近）；站 60 秒以上，傾斜一直是 0.0°。KP 改 60：9 秒時傾斜 1.5°，還站得住；KP 改 6：2 秒 13°、5 秒內倒下 |
| `train.py` 30 分鐘 | 30.3 萬步（每秒約 170 步，含 PPO 更新）。前 10 萬步幾乎只是站著；11–22 萬步開始往前走、跌倒變多，`ep_len_mean` 1200 → 580、`ep_rew_mean` 一起掉；之後回升到 713 / 1606。每步平均分數一路從 1.93 升到 2.26 |
| TensorBoard | `tensorboard --logdir runs` 開 `localhost:6006`，訓練中就看得到曲線 |
| `play.py`（帶雜訊） | 30 萬步：20 秒平均往前 1.34 m（0.17–2.03 m）、4 次都沒跌倒，前傾碎步；5 萬步只在原地抖（−0.09 m） |
| `play.py` 改 `deterministic=True` | 30 萬步只會站著、往後滑 0.10 m：學到的走法要靠雜訊當踏步 → `play.py` 預設帶雜訊 |

`train.py` 裡關掉「把物理結果寫回 USD」（`/physics/updateToUsd`）：物理結果一樣，速度從每秒 75 步變 187 步。
