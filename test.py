import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # 触发 __init__.py 里的 register
import numpy as np
import types, os
from util.test_fps import test_env_fps
from util.plot_result import plot_ego_speed_history, plot_all_speed_history, plot_warmup_avg_speed

from rl.algos.ppo.ppo import PPO
from rl.algos.sac.sac import SAC

def load_model(algo: str, model_path: str, env):
    algo = algo.lower()
    if algo == "ppo":
        model = PPO.load(model_path, env=env)
    elif algo == "sac":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"未知算法类型: {algo}")
    return model


def record_episode_video(
    model,
    model_path: str,
    algo: str,
    env_config: dict,
    episode_seed: int,
    episode_idx: int,
):
    video_folder = model_path
    name_prefix = f"{algo}_ep{episode_idx}"

    # 专门用于录制的环境：render_mode="rgb_array" + RecordVideo
    base_env = gym.make(
        "multi-lane-custom-v0",
        render_mode="rgb_array",
        config=env_config,
    )
    env = RecordVideo(
        base_env,
        video_folder=video_folder,
        episode_trigger=lambda ep_id: True,  # 只跑一个 episode，第一条就录
        name_prefix=name_prefix,
    )

    obs, _ = env.reset(seed=episode_seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    print(f"[VIDEO] 已录制 episode {episode_idx} 的视频到: {video_folder}")


def main(model_path: str, model_name: str, algo: str, episodes: int, record_episode: int | None = None):
    # 日志文件：模型所在目录下的 txt
    log_path = os.path.join(model_path, f"eval_{algo.lower()}.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    env_config = {
        # 这里可以覆盖 default_config 里定义的任何键
        "policy_frequency": 10,  # [Hz]
        "screen_width": 1200,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": 'all',  # True 时记录并显示ego轨迹, all 时记录所有车辆轨迹
        "warmup_render": False,      # True 时在 reset 期间也渲染 warmup 画面
        "real_time_rendering": False,  # True 时在 step 期间渲染时加 sleep，变成肉眼可看速度
        "observation": {
            "type": "Kinematics",
            "normalize": False,
            "include_time": True,  # 在观测中加入当前时间
            "time_range": [0.0, 30.0],
        },
    }
    env = gym.make(
        "multi-lane-custom-v0",
        render_mode="human",
        config=env_config,
    )

    # 记录每一步的 reward 各项分量
    seed_base = 42
    obs, _ = env.reset(seed=seed_base)
    plot_warmup_avg_speed(env)

    if env.unwrapped.render_mode is not None:
        # 定义一个“假车”，把摄像头锁在中心点
        class Dummy:
            def __init__(self, pos):
                self.position = np.array(pos, dtype=float)
        env.render()    # 先 render 一帧，env 内部会创建 env.viewer
        env.unwrapped.viewer.observer_vehicle = Dummy([env.unwrapped.config["road_length"]/2, 5.0])

    # 加载模型
    model = load_model(algo, os.path.join(model_path, model_name), env)

    # reward key 列表，保持与 MultiLaneEnv._rewards 一致
    reward_keys = [
        "collision_reward",
        "progress_reward",
        "comfort_reward",
        "lane_change_reward",
        "punctual_reward",
        "on_road_reward",
    ]
    log("=" * 80)
    log(f"Eval model: {model_path}")
    log(f"Algo      : {algo}")
    log(f"Episodes  : {episodes}")
    log("=" * 80)

    # 用于统计均值
    episode_lengths = []
    episode_total_rewards = []
    agg_components = {k: 0.0 for k in reward_keys}
    arrived_count = 0
    arrival_times = []

    for ep in range(1, episodes+1):
        # 每个 episode 重置环境和统计量
        episode_seed = seed_base + ep
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        ep_total_reward = 0.0
        ep_components = {k: 0.0 for k in reward_keys}

        while not (terminated or truncated):
            # 用训练好的模型选择动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 从 env 中取出刚刚这个 step 的 reward 分量
            r_dict = getattr(env.unwrapped, "_last_weighted_rewards", None)
            if r_dict is not None:
                for k in reward_keys:
                    ep_components[k] += float(r_dict.get(k, 0.0))
            ep_total_reward += float(reward)
            step_count += 1
            if env.unwrapped.render_mode is not None:
                env.render() 
        # Episode 结束，判断是否成功到达
        base_env = env.unwrapped
        crashed = getattr(base_env.vehicle, "crashed", False)
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time = getattr(base_env, "_arrival_time", None)

        # 打印本 episode 结果
        reason = "unknown"
        if truncated:
            reason = "truncated(time limit)"
        else:
            # terminated
            if crashed:
                reason = "terminated(crash)"
            elif arrived:
                reason = "terminated(goal)"
            else:
                reason = "terminated(other)"
        # 统计到全局
        episode_lengths.append(step_count)
        episode_total_rewards.append(ep_total_reward)
        for k in reward_keys:
            agg_components[k] += ep_components[k]
        if arrived and (arrival_time is not None):
            arrived_count += 1
            arrival_times.append(float(arrival_time))

        # ---- 打印 / 写日志：单个 episode 结果 ----
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  length (steps) : {step_count}")
        log(f"  total reward   : {ep_total_reward:.4f}")
        log(f"  terminated info: {reason}")
        log("  reward components (sum over episode):")
        for k in reward_keys:
            log(f"    {k:18s}: {ep_components[k]: .6f}")
        # 成功到达目标时，打印到达时间
        if arrived and arrival_time is not None:
            log(f"  ARRIVED at t = {arrival_time:.3f} s")
        # 如需画速度轨迹
        if base_env.config["show_trajectories"]:
            if base_env.config["show_trajectories"] == 'all':
                plot_all_speed_history(env)
            else:
                plot_ego_speed_history(env)
        # 如果需要录制这一条 episode 的视频
        if record_episode is not None and ep == record_episode:
            record_episode_video(
                model=model,
                model_path=model_path,
                algo=algo,
                env_config=env_config,
                episode_seed=episode_seed,
                episode_idx=ep,
            )
    
    # ====== 统计所有 episode 的均值并打印 ======
    n = episodes
    mean_len = float(np.mean(episode_lengths)) if n > 0 else 0.0
    mean_total_rew = float(np.mean(episode_total_rewards)) if n > 0 else 0.0
    log("=" * 80)
    log("Summary over all episodes:")
    log(f"  episodes          : {n}")
    log(f"  mean length       : {mean_len:.3f} steps")
    log(f"  mean total reward : {mean_total_rew:.6f}")
    log("  mean reward components (per episode):")
    for k in reward_keys:
        mean_k = agg_components[k] / n
        log(f"    {k:18s}: {mean_k: .6f}")
    arrive_rate = arrived_count / n
    log(f"  arrival rate      : {arrive_rate * 100:.2f}%")
    if arrived_count > 0:
        mean_arrival_time = float(np.mean(arrival_times))
        log(f"  mean arrival time : {mean_arrival_time:.3f} s (over {arrived_count} success episodes)")
    else:
        log("  mean arrival time : N/A (no successful episodes)")
    log("=" * 80)

    log_file.close()
    env.close()


if __name__ == "__main__":
    # main(
    #     model_path="./models/ppo_1e6",
    #     model_name="best_model.zip",
    #     algo="ppo",
    #     episodes=30,
    #     record_episode=3,
    # )
    main(
        model_path="./models/sac_5e6",
        model_name="best_model.zip",
        algo="sac",
        episodes=30,
        record_episode=None,
    )