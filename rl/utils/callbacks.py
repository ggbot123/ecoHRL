from stable_baselines3.common.callbacks import BaseCallback

class RewardComponentsTensorboardCallback(BaseCallback):
    """
    每个 rollout step，从 env 的 info["reward_components"] 中取出各分项奖励（加权后），
    在 TensorBoard 上记录它们的平均值曲线。

    对向量环境 (DummyVecEnv)：
        - self.locals["infos"] 是一个长度为 num_envs 的 info 列表
        - 我们对其中的 reward_components 做 env-维度平均
    """

    def __init__(self, log_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        # 和你的 _rewards 字典里的 key 对齐
        self.reward_keys = [
            "collision_reward",
            "progress_reward",
            "comfort_reward",
            "lane_change_reward",
            "punctual_reward",
            "on_road_reward",
        ]

    def _on_step(self) -> bool:
        # 每 log_freq 次 step 记录一次，避免太密
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos")
        if infos is None:
            return True

        # infos 是 list[dict]（每个子环境一个）
        sums = {k: 0.0 for k in self.reward_keys}
        count = 0

        for info in infos:
            rc = info.get("reward_components")
            if rc is None:
                continue
            for k in self.reward_keys:
                sums[k] += float(rc.get(k, 0.0))
            count += 1

        if count == 0:
            return True

        # env 之间做个平均
        for k in self.reward_keys:
            mean_val = sums[k] / count
            # 写入 TensorBoard，tag 自己起个好分组名字
            self.logger.record(f"reward_components/{k}", mean_val)

        # 继续训练
        return True