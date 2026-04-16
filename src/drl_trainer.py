"""
drl_trainer.py — Thuật toán PPO huấn luyện Agent cho Virtual Gym
"""

import os
import json
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from virtual_gym import VirtualStockEnv
except ImportError:
    PPO = None
    VirtualStockEnv = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "reports", "ppo_stock_agent.zip")
LOG_PATH = os.path.join(DATA_DIR, "reports", "json", "drl_training_status.json")

# Global training thread tracking
TRAINING_THREAD = None
IS_TRAINING = False
STATUS_LOCK = threading.Lock()

def update_status(status: str, progress: float, details: str = ""):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    data = {"status": status, "progress": progress, "details": details, "updated_at": datetime.now().isoformat()}
    with STATUS_LOCK:
        with open(LOG_PATH, "w") as f:
            json.dump(data, f)

def start_training(ticker: str = "VNM", episodes: int = 50, initial_capital: float = 9000000):
    global IS_TRAINING, TRAINING_THREAD
    
    if IS_TRAINING:
        return {"ok": False, "error": "Đang có tiến trình training DRL chạy"}
    
    if PPO is None:
        return {"ok": False, "error": "Thiếu thư viện stable-baselines3 / gymnasium. Hãy chạy pip install stable-baselines3 gymnasium"}
    
    def _train():
        global IS_TRAINING
        IS_TRAINING = True
        update_status("Khởi tạo Môi trường", 0.0)
        
        try:
            env = VirtualStockEnv(ticker=ticker, initial_capital=initial_capital)
            
            # Load existing model if exists, otherwise create new
            if os.path.exists(MODEL_PATH):
                model = PPO.load(MODEL_PATH, env=env)
                update_status("Tiếp tục Training Model cũ", 5.0)
            else:
                model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64)
                update_status("Khởi tạo Model mới", 5.0)
            
            total_timesteps = env.max_steps * episodes
            
            # Fake progress loop over chunks to update dashboard
            timesteps_per_update = env.max_steps * 5
            total_updates = total_timesteps // timesteps_per_update
            
            for i in range(total_updates):
                if not IS_TRAINING:
                    break
                
                model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)
                
                prog = 5.0 + (i / total_updates) * 90.0
                obs, _ = env.reset()
                
                # Check performance via a quick rollout
                d_nav = env.cash + env.shares * (obs[0] * env.df.iloc[env.current_step]["ema20"]) # Approx nav
                
                update_status("Đang cày cuốc (Training)", round(prog, 1), f"Episode Chunk {i+1}/{total_updates} | Est. NAV: {d_nav:,.0f} VND")
                
            model.save(MODEL_PATH)
            update_status("Hoàn tất Training", 100.0, "Đã lưu Model PPO vào data/reports/ppo_stock_agent.zip")
            
        except Exception as e:
            logger.error(str(e))
            update_status("Lỗi Training", 0.0, str(e))
        finally:
            IS_TRAINING = False

    TRAINING_THREAD = threading.Thread(target=_train, daemon=True)
    TRAINING_THREAD.start()
    return {"ok": True, "message": "Bắt đầu Background DRL Training"}

def stop_training():
    global IS_TRAINING
    IS_TRAINING = False
    return {"ok": True, "message": "Đã ra lệnh dừng"}

def get_status():
    if not os.path.exists(LOG_PATH):
        return {"status": "Chưa có tiến trình", "progress": 0.0, "details": ""}
    with STATUS_LOCK:
        with open(LOG_PATH, "r") as f:
            return json.load(f)

if __name__ == "__main__":
    print(start_training("VNM", episodes=2))
