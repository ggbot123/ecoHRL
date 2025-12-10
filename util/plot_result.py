import matplotlib.pyplot as plt

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
    vehs = env.unwrapped.road.vehicles  # 所有车辆字典
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
