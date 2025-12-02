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