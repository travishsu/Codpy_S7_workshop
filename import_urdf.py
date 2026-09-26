# import_urdf.py — 把 URDF 轉成 USD（Isaac Sim 6.0 的新 API）
#   python import_urdf.py              自動找這個資料夾裡唯一的 .urdf
#   python import_urdf.py xxx.urdf     指定檔案
import glob, os, shutil, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

urdfs = sys.argv[1:] or glob.glob("**/*.urdf", recursive=True)
if len(urdfs) != 1:
    print("\n".join(urdfs))
    sys.exit(f"找到 {len(urdfs)} 個 .urdf：請用 python import_urdf.py <檔案> 指定一個")
urdf = os.path.abspath(urdfs[0])
name = os.path.splitext(os.path.basename(urdf))[0]
shutil.rmtree(os.path.join("usd", name), ignore_errors=True)   # 不刪的話第二次會變 xxx_01

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.urdf")
app.update()
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

cfg = URDFImporterConfig(
    urdf_path=urdf,
    usd_path=os.path.abspath("usd"),  # 輸出資料夾 → usd/<名字>/<名字>.usda
    fix_base=False,                   # 會走路：底座不固定
    merge_fixed_joints=False,
    collision_from_visuals=False,     # 用 URDF 自己的 collision
    collision_type="Convex Hull",     # 網格碰撞 → 凸包
    allow_self_collision=True,        # 6.0.1 只寫給 Newton；PhysX 預設就會自碰撞
    link_density=None,                # 質量用 URDF 的，不要自己算
    run_asset_transformer=False,      # 輸出單一 .usda
)
usd = URDFImporter(cfg).import_urdf()
with open("robot_usd.txt", "w") as f:  # 記下來：stand / train / play 預設讀它
    f.write(usd)
print("wrote", usd, flush=True)
os._exit(0)                           # Windows 上 app.close() 可能報錯，直接結束
