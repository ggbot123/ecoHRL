import gymnasium as gym
import scenarios.multi_lane  # 触发你在 __init__.py 里的 register
import numpy as np
import time
from util.test_fps import test_env_fps

np.random.seed(42)  # 全局随机数生成器，为了可复现的随机场景
env = gym.make(
    "multi-lane-custom-v0",
    render_mode="human",
    # render_mode=None,
    config={
        # 这里可以覆盖 default_config 里定义的任何键
        "screen_width": 1200,
        "screen_height": 300,
        "scaling": 2.0,              # 越小越“缩放出去”，视野越大
        "centering_position": [0.5, 0.5],  # 观察点放在屏幕中心
        "show_trajectories": False,      # True 时记录并显示车辆轨迹
        "warmup_render": False,          # True 时在 reset 期间也渲染 warmup 画面
        "real_time_rendering": True,     # True 时在 step 期间渲染时加 sleep，变成肉眼可看速度
    },
)

# 初始化
seed = np.random.randint(0, 1e7)
obs, info = env.reset(seed=seed)

# 定义一个“假车”，把摄像头锁在中心点
class Dummy:
    def __init__(self, pos):
        self.position = np.array(pos, dtype=float)
env.render()    # 先 render 一帧，env 内部会创建 env.viewer
env.unwrapped.viewer.observer_vehicle = Dummy([250.0, 5.0])

for _ in range(1000):
    # action = env.action_space.sample()
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    img = env.render()
    if terminated or truncated:
        obs, info = env.reset(seed=np.random.randint(0, 1e7))