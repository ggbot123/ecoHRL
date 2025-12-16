import os
import matplotlib.pyplot as plt
import numpy as np

def plot_ego_speed_history(env):
    ego = env.unwrapped.vehicle          # ego 车对象
    hist = list(reversed(ego.history))   # deque -> list
    speeds = [v.speed for v in hist]
    dt = 1.0 / env.unwrapped.config["simulation_frequency"]
    times = [i * dt for i in range(len(speeds))]
    plt.plot(times, speeds)
    plt.xlabel("Time [s]")
    plt.ylabel("Ego speed [m/s]")
    plt.grid(True)
    plt.show()

def plot_all_speed_history(env):
    dt = 1.0 / env.unwrapped.config["simulation_frequency"]
    vehs = env.unwrapped.road.vehicles  # 所有车辆
    for v in vehs:
        hist = list(reversed(v.history))
        speeds = [t.speed for t in hist]
        times = [i * dt for i in range(len(speeds))]
        if v == env.unwrapped.vehicle:
            plt.plot(times, speeds, color='r')
        else:
            plt.plot(times, speeds, color='b')

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.grid(True)
    plt.show()


def plot_warmup_avg_speed(env, show=True, save_path=None):
    base_env = env.unwrapped
    times = getattr(base_env, "_warmup_times", None)
    avg_speeds = getattr(base_env, "_warmup_avg_speeds", None)
    if times is None or avg_speeds is None:
        raise RuntimeError("env 中没有 warmup 统计信息，请确认已经执行过第一次 reset。")
    plt.figure()
    plt.plot(times, avg_speeds)
    plt.xlabel("Time [s]")
    plt.ylabel("Average speed [m/s]")
    plt.title("Warmup average speed vs time")
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def save_speed_acc_curves(env, ep_idx: int, model_path: str):
    """
    在 show_trajectories 打开的情况下，将当前 episode 的
    车速曲线、加速度曲线和所在车道随时间变化曲线保存到：
        model_path/speed_curve/epXXX_speed.png
        model_path/acc_curve/epXXX_acc.png
        model_path/lane_curve/epXXX_lane.png

    - show_trajectories == 'all'：一张图上画所有车辆（ego 为红色，其它车辆为蓝色）
    - show_trajectories == True：只画 ego 车辆
    - show_trajectories == False：不做任何事
    """
    base_env = env.unwrapped
    road = base_env.road

    show_mode = base_env.config.get("show_trajectories", False)
    if not show_mode:
        # 未开启轨迹记录，直接返回
        return

    # 创建保存目录
    speed_dir = os.path.join(model_path, "speed_curve")
    acc_dir = os.path.join(model_path, "acc_curve")
    lane_dir = os.path.join(model_path, "lane_curve")
    os.makedirs(speed_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(lane_dir, exist_ok=True)

    speed_path = os.path.join(speed_dir, f"ep{ep_idx:03d}_speed.png")
    acc_path = os.path.join(acc_dir, f"ep{ep_idx:03d}_acc.png")
    lane_path = os.path.join(lane_dir, f"ep{ep_idx:03d}_lane.png")

    # 时间步长按 simulation_frequency 计算（与 history 记录频率一致）
    dt = 1.0 / float(base_env.config["simulation_frequency"])

    # 根据 show_trajectories 的取值决定绘制哪些车辆
    if show_mode == "all":
        vehicles = list(road.vehicles)
        title_prefix = "All vehicles"
    else:
        vehicles = [base_env.vehicle]
        title_prefix = "Ego"

    # --------- 速度曲线 --------- #
    plt.figure()
    for v in vehicles:
        hist = list(reversed(getattr(v, "history", [])))
        if not hist:
            continue
        speeds = np.asarray([snap.speed for snap in hist], dtype=float)
        if speeds.size == 0:
            continue
        t = np.arange(speeds.size, dtype=float) * dt
        if v is base_env.vehicle:
            plt.plot(t, speeds, color="r", label="ego")
        else:
            plt.plot(t, speeds, color="b", alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title(f"{title_prefix} Speed vs Time (ep {ep_idx})")
    plt.grid(True)
    if show_mode == "all":
        plt.legend()
    plt.tight_layout()
    plt.savefig(speed_path)
    plt.close()

    # --------- 加速度曲线（由速度数值微分算出） --------- #
    plt.figure()
    for v in vehicles:
        hist = list(reversed(getattr(v, "history", [])))
        if not hist:
            continue
        speeds = np.asarray([snap.speed for snap in hist], dtype=float)
        if speeds.size < 2:
            continue
        # 数值微分：a_t ≈ (v_t - v_{t-1}) / dt
        accs = np.diff(speeds) / dt          # 长度 N-1
        t_acc = np.arange(accs.size, dtype=float) * dt

        if v is base_env.vehicle:
            plt.plot(t_acc, accs, color="r", label="ego")
        else:
            plt.plot(t_acc, accs, color="b", alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title(f"{title_prefix} Acceleration vs Time (ep {ep_idx})")
    plt.grid(True)
    if show_mode == "all":
        plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    # --------- 车道随时间变化曲线 --------- #
    def _get_lane_id(snap):
        li = getattr(snap, "lane_index", None)
        if li is None:
            return np.nan
        # highwayEnv 风格：lane_index = (from, to, lane_id)
        try:
            if isinstance(li, (tuple, list)) and len(li) >= 3:
                return float(li[2])
            # 其他情况尝试直接转为数值
            return float(li)
        except Exception:
            return np.nan

    plt.figure()
    for v in vehicles:
        hist = list(reversed(getattr(v, "history", [])))
        if not hist:
            continue
        lane_ids = np.asarray([_get_lane_id(snap) for snap in hist], dtype=float)
        if lane_ids.size == 0:
            continue
        t_lane = np.arange(lane_ids.size, dtype=float) * dt

        if v is base_env.vehicle:
            plt.step(t_lane, lane_ids, where="post", color="r", label="ego")
        else:
            plt.step(t_lane, lane_ids, where="post", color="b", alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Lane ID")
    plt.title(f"{title_prefix} Lane vs Time (ep {ep_idx})")
    plt.grid(True)
    if show_mode == "all":
        plt.legend()
    plt.tight_layout()
    plt.savefig(lane_path)
    plt.close()

