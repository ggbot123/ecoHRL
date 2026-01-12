import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def save_goal_snapshot(env, runner, ep_idx: int, step: int, model_dir: str, prev_goal_phys=None, intrinsic_reward=None, folder_name="goal_distribution"):
    """
    保存 HIRO Goal 可视化快照 (Vector Graphics version).
    使用 Matplotlib 直接绘制道路和车辆，获得清晰的矢量图/高分辨率位图，
    """
    import matplotlib.transforms as transforms
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # 1. Directory Structure: separated by episode
    # Clear output folder once at the beginning of a run
    base_debug_dir = os.path.join(model_dir, "debug", folder_name)
    if int(ep_idx) == 1 and int(step) == 0:
        if os.path.exists(base_debug_dir):
            shutil.rmtree(base_debug_dir)
    debug_dir = os.path.join(base_debug_dir, f"ep{ep_idx:03d}")
    os.makedirs(debug_dir, exist_ok=True)
    
    base_env = env.unwrapped
    road = base_env.road
    ego = base_env.vehicle
    
    if runner.goal_phys is None:
        return

    # 获取感知范围
    p_dist = getattr(base_env, "PERCEPTION_DISTANCE", 200.0)
    if p_dist is None: p_dist = 200.0
    p_dist = float(p_dist)
    
    # 获取范围内车辆
    # close_vehicles_to 返回按距离排序的车辆列表 (不含 ego)
    neighbors = road.close_vehicles_to(ego, p_dist)
    
    # 确定哪些是 "Local Prob" (Observation 内的车辆)
    # runner.n_veh_local 是观察空间中包含的邻车数量
    n_local = getattr(runner, "n_veh_local", 0)
    local_neighbors_set = set(neighbors[:n_local])
    
    # 绘图列表：ego + neighbors
    # 注意：绘制顺序影响遮挡，这里不严格区分，因为大家都在车道上
    all_draw_vehs = [ego] + neighbors

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # 3. Draw Road Planes
    lanes = road.network.lanes_list()
    ys = []
    
    for lane in lanes:
        x0, y0 = lane.start
        x1, y1 = lane.end
        w = lane.width
        
        heading = lane.heading_at(0)
        c, s = np.cos(heading), np.sin(heading)
        normal = np.array([-s, c])
        
        p0 = lane.start - normal * w / 2
        p1 = lane.start + normal * w / 2
        p2 = lane.end + normal * w / 2
        p3 = lane.end - normal * w / 2
        
        poly = patches.Polygon([p0, p1, p2, p3], closed=True, facecolor='#666666', edgecolor='none', zorder=0)
        ax.add_patch(poly)
        
        types = [str(t) for t in lane.line_types]
        
        def draw_line(pa, pb, ltype):
            if 'NONE' in ltype: return
            style = 'solid' if 'CONTINUOUS' in ltype else 'dashed'
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color='white', linestyle=style, linewidth=1, zorder=1)
            
        draw_line(p0, p3, types[0])
        draw_line(p1, p2, types[1])
        
        ys.append(y0)
        ys.append(y1)
        
    # 4. Draw Vehicles
    # 动态 Colorbar Range: 使用 HIRO High-Level Output 绝对速度范围 [0, speed_limit]
    # 索引获取: init_kinematics_meta 中 keep = ("x", "y", "vx", "vy")
    sx, sy, svx, svy = runner.ego_start[:4]
    
    # 获取速度上限 (默认 30 m/s，如果 config 中未定义)
    speed_limit = float(base_env.config.get("speed_limit", 30.0))
    norm = mcolors.Normalize(vmin=0.0, vmax=speed_limit)
    cmap = cm.get_cmap('jet') # 使用区分度高的色谱
    
    for v in all_draw_vehs:
        # Color Logic
        edge_c = 'black'
        lw = 1
        z = 5
        alpha_v = 0.9
        
        if v is ego:
            # Ego: Color by speed, Red Edge (thin)
            color = cmap(norm(v.speed))
            edge_c = 'red'
            lw = 1
            z = 6
        elif v in local_neighbors_set:
            # Observed Neighbor: Color by speed
            color = cmap(norm(v.speed))
        elif getattr(v, "crashed", False):
            color = 'black'
        else:
            # Unobserved: Gray
            color = '#E0E0E0'
        
        l, w = v.LENGTH, v.WIDTH
        rect = patches.Rectangle((-l/2, -w/2), l, w, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha_v, zorder=z)
        
        t = transforms.Affine2D().rotate(v.heading).translate(v.position[0], v.position[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # 5. Draw Goal & Range
    # Unpack first 4 elements: x, y, vx, vy
    gx, gy, gvx, gvy = runner.goal_phys[:4]
    
    # Draw Goal (Dot with color by Absolute Speed)
    goal_color = cmap(norm(gvx))
    ax.scatter([gx], [gy], c=[goal_color], marker='o', s=50, linewidth=1.5, edgecolors='white', zorder=10)
    
    # Draw Previous Goal (Transparent Dot)
    if prev_goal_phys is not None and len(prev_goal_phys) >= 4:
         px, py, pvx, pvy = prev_goal_phys[:4]
         if pvx != 0 or px != 0: # 简单过滤初始全0的情况
            p_color = cmap(norm(pvx))
            ax.scatter([px], [py], c=[p_color], marker='o', s=50, linewidth=1.5, edgecolors='white', zorder=9, alpha=0.5)

    # Calculate Average Acceleration Req
    # Acc = Delta V / Duration
    pol_freq = float(base_env.config.get("policy_frequency", 10.0))
    dt = 1.0 / pol_freq
    hi_steps = getattr(runner, "hi", 10)
    duration = max(hi_steps * dt, 1e-3)
    avg_acc = (gvx - svx) / duration
    
    # Draw Range Box
    x_range = runner.norm_ranges[0]
    y_range = runner.norm_ranges[1]
    box_min_x = sx + x_range[0]
    box_max_x = sx + x_range[1]
    box_min_y = sy + y_range[0]
    box_max_y = sy + y_range[1]
    w_box = box_max_x - box_min_x
    h_box = box_max_y - box_min_y
    rect_range = patches.Rectangle((box_min_x, box_min_y), w_box, h_box,
                                   linewidth=2, edgecolor='lime', facecolor='none', linestyle='--', zorder=8)
    ax.add_patch(rect_range)

    # 6. View Settings
    x_min = ego.position[0] - p_dist
    x_max = ego.position[0] + p_dist
    ax.set_xlim(x_min, x_max)
    
    if ys:
        mean_y = np.mean(ys)
        ax.set_ylim(mean_y - 12, mean_y + 12)
    else:
        ax.set_ylim(-10, 10)
        
    ax.invert_yaxis()  # HighwayEnv Y轴正方向向下，需翻转 Matplotlib 默认行为
    ax.set_aspect('equal')
    ax.axis('off')

    # Add Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label('Speed [m/s]', fontsize=10)
    
    # Add Title with Goal Info
    title_str = f"Ep {ep_idx} Step {step}\nGoal V: {gvx:.1f} m/s | Avg Acc Req: {avg_acc:.2f} m/s²"
    if intrinsic_reward is not None:
        title_str += f"\nLast Int. Reward: {intrinsic_reward:.4f}"
    plt.title(title_str)
    
    save_path = os.path.join(debug_dir, f"step{step:05d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
