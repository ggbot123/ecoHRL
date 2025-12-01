import time
import gymnasium as gym
import custom_env   # 确保环境注册好


def test_env_fps(env, max_steps=1000, render=False):
    """
    简单测一下一个 episode 中的 FPS。

    参数
    ----
    env : gymnasium.Env
        已经创建好的环境（包含 config）
    max_steps : int
        最多跑多少个 env.step（防止 episode 太长）
    render : bool
        是否在测试时调用 env.render()（会大幅降低 FPS）
    """
    obs, info = env.reset()

    start_wall = time.time()
    step_count = 0

    terminated = False
    truncated = False

    while not (terminated or truncated) and step_count < max_steps:
        action = env.action_space.sample()  # 或者用你的策略
        obs, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        step_count += 1

    end_wall = time.time()
    elapsed = end_wall - start_wall

    if elapsed <= 0:
        print("时间太短，测不出 FPS")
        return

    # 以“决策步”为单位的 FPS
    step_fps = step_count / elapsed

    # 再换算成 simulation step 的 FPS（如果有这两个配置）
    cfg = getattr(env.unwrapped, "config", {})
    sim_freq = cfg.get("simulation_frequency", None)
    pol_freq = cfg.get("policy_frequency", None)

    if sim_freq is not None and pol_freq is not None and pol_freq > 0:
        frames_per_step = sim_freq / pol_freq
        sim_fps = step_fps * frames_per_step
        print(f"Steps: {step_count}, Time: {elapsed:.3f}s")
        print(f"决策步 FPS: {step_fps:.1f} step/s")
        print(f"仿真步 FPS: {sim_fps:.1f} sim_step/s "
              f"(simulation_frequency={sim_freq}, policy_frequency={pol_freq})")
    else:
        print(f"Steps: {step_count}, Time: {elapsed:.3f}s")
        print(f"决策步 FPS: {step_fps:.1f} step/s（没找到 sim/policy freq，无法换算 sim_step）")
