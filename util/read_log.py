from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# 你的 log 文件所在的文件夹路径
log_root = "../logs/hiro_1e6_lane_random/hiro_low/hiro_low_1"
log_path = f"{log_root}/events.out.tfevents.1765800090.BF-202209291636.28016.1"

# 加载日志，size_guidance=0 表示不限制大小
event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
event_acc.Reload()

# 获取指定标签的数据 (例如 'rollout/ep_rew')
tags = event_acc.Tags()['scalars']
print("Available tags:", tags)

data = []
for tag in tags:
    events = event_acc.Scalars(tag)
    for event in events:
        data.append({
            "step": event.step,
            "wall_time": event.wall_time,
            "value": event.value,
            "tag": tag
        })

# 转为 DataFrame 并保存
df = pd.DataFrame(data)
df.to_csv(f"{log_root}/full_log_data.csv", index=False)
print("全量数据已导出！")