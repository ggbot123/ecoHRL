import numpy as np
from scipy.optimize import minimize
from custom_env.vehicle.kinematics import Vehicle
from custom_env.road.lane import StraightLane
import copy

class MPCController:
    """
    Model Predictive Controller (MPC) for HighwayEnv.
    Optimizes for intrinsic reward (HIRO goal tracking), comfort, and progress,
    subject to collision avoidance constraints.
    """
    def __init__(self, 
                 env, 
                 horizon=10, 
                 dt=0.1, 
                 weights=None, 
                 intrinsic_norm_ranges=None,
                 intrinsic_coef=1.0,
                 intrinsic_weights=None):
        self.env = env
        self.horizon = horizon
        self.dt = dt
        
        # Default weights (tune these to match HIRO lower-level training)
        # Weights: [intrinsic, comfort, progress]
        self.weights = weights if weights is not None else {'intrinsic': 1.0, 'comfort': 1.0, 'progress': 1.0}
        
        # Intrinsic reward parameters
        self.intrinsic_norm_ranges = intrinsic_norm_ranges
        self.intrinsic_coef = intrinsic_coef
        self.intrinsic_weights = intrinsic_weights if intrinsic_weights is not None else np.array([1.0, 2.0, 8.0, 1.0])

        # Vehicle parameters (approximate)
        self.LENGTH = 5.0
        self.WIDTH = 2.0
        
        # Bounds for optimization variables (acceleration, steering)
        self.acc_max = 6.0
        self.steering_max = np.pi / 4

    def predict_trajectory(self, state, actions):
        """
        Rollout the kinematic bicycle model.
        state: [x, y, vx, vy, heading] (simplified)
        actions: list of [acc, steering]
        """
        traj = [state]
        x, y, vx, vy, psi = state
        
        # Simplified kinematic model
        # Assuming steering directly affects heading rate for simplicity or use bicycle model
        # HighwayEnv Vehicle dynamics:
        # dx/dt = v * cos(psi + beta)
        # dy/dt = v * sin(psi + beta)
        # dv/dt = a
        # dpsi/dt = v / L * sin(beta)  (if steering is beta)
        # Here we approximate action as [acceleration, steering_angle]
        
        current_state = np.array([x, y, vx, vy, psi])
        
        for acc, steering in actions:
            # Clip actions
            acc = np.clip(acc, -self.acc_max, self.acc_max)
            steering = np.clip(steering, -self.steering_max, self.steering_max)
            
            # Update state (Euler integration)
            v = np.sqrt(vx**2 + vy**2)
            
            # Simple Kinematic Bicycle Model
            # beta = arctan(0.5 * tan(steering))
            beta_slip = np.arctan(0.5 * np.tan(steering))
            
            dx = v * np.cos(psi + beta_slip) * self.dt
            dy = v * np.sin(psi + beta_slip) * self.dt
            dv = acc * self.dt
            # Match Env dynamics: dpsi = v * sin(beta) / (L/2) * dt
            dpsi = (v / (self.LENGTH / 2)) * np.sin(beta_slip) * self.dt
            
            x += dx
            y += dy
            v = np.clip(v + dv, 0.0, 40.0) # Clip speed to range [0, MAX_SPEED] roughly
            psi += dpsi
            
            # Update vx, vy for next step
            vx = v * np.cos(psi)
            vy = v * np.sin(psi)
            
            traj.append(np.array([x, y, vx, vy, psi]))
            
        return np.array(traj)

    def cost_function(self, actions_flat, start_state, goal_phys, lane_center_ys, neighbors, steps_to_goal):
        """
        Calculate total cost for a sequence of actions.
        actions_flat: flattened array of actions [acc1, str1, acc2, str2, ...]
        steps_to_goal: int, steps remaining to reach the goal (intrinsic reward evaluation point)
        """
        actions = actions_flat.reshape(-1, 2)
        traj = self.predict_trajectory(start_state, actions)
        
        total_cost = 0.0
        
        # 1. Intrinsic Reward Cost (Distance to Goal at fixed time step)
        # We want to minimize distance to goal at the specific target time step
        eval_step_idx = int(min(len(traj) - 1, steps_to_goal))
        if eval_step_idx < 1: eval_step_idx = 1
        
        eval_state = traj[eval_step_idx]
        
        ego_state_phys = eval_state[:4] # [x, y, vx, vy]
        
        # Normalized distance calculation (similar to utils.intrinsic_reward_l2)
        # But here we assume `intrinsic_norm_ranges` is standard deviation or max range
        # intrinsic_reward = - || (s' - s_start) - (g - s_start) ||_W
        # Equivalent to minimizing || s' - g ||_W
        # goal_phys is absolute goal state
        
        diff = ego_state_phys - goal_phys # [dx, dy, dvx, dvy]
        
        # Normalize diff if ranges provided
        if self.intrinsic_norm_ranges is not None:
             # ranges format: [[x_min, x_max], [y_min, y_max], ...]
             # simple normalization by span? or std?
             # utils.py uses: (val) / (range_max - range_min) * 2 or similar
             # Here we simplified: cost weighted squared error
             pass

        # Weighted squared error
        dist_sq = np.sum(self.intrinsic_weights * (diff**2))
        cost_intrinsic = dist_sq * self.intrinsic_coef
        
        # 2. Comfort Cost (Acceleration and Jerk)
        # Only accumulate cost up to the evaluation step
        acc_seq = actions[:eval_step_idx, 0]
        # steering_seq = actions[:, 1]
        
        cost_comfort = np.sum(acc_seq**2)
        
        # 3. Progress Cost (Maximize longitudinal progress)
        # Minimize -x at evaluation step
        cost_progress = -eval_state[0]
        
        # 4. Collision Cost (Soft Constraint or Hard Penalty)
        # Check distance to neighbors at each step UNTIL evaluation step
        cost_collision = 0.0
        min_safe_dist = 6.0 # meters
        
        for t in range(1, eval_step_idx + 1):
            ego_x, ego_y = traj[t, 0], traj[t, 1]
            for v_neighbor in neighbors:
                # Simple prediction for neighbor: constant velocity or current state
                # Here assuming constant velocity for neighbors
                # Neighbor state: [x, y, vx, vy]
                n_x = v_neighbor.position[0] + v_neighbor.speed * np.cos(v_neighbor.heading) * (t * self.dt)
                n_y = v_neighbor.position[1] + v_neighbor.speed * np.sin(v_neighbor.heading) * (t * self.dt)
                
                d = np.sqrt((ego_x - n_x)**2 + (ego_y - n_y)**2)
                if d < min_safe_dist:
                    cost_collision += 1000.0 * (min_safe_dist - d)**2
                    
        total_cost = (self.weights['intrinsic'] * cost_intrinsic +
                      self.weights['comfort'] * cost_comfort + 
                      self.weights['progress'] * cost_progress + 
                      cost_collision)
                      
        return total_cost

    def plan(self, obs, goal_phys, steps_to_goal):
        """
        Plan optimal action sequence.
        obs: current observation (extract ego state)
        goal_phys: absolute goal state [x, y, vx, vy]
        steps_to_goal: steps remaining until goal state is expected
        """
        ego = self.env.vehicle
        neighbors = self.env.road.vehicles # Filter for close ones
        neighbors = [v for v in neighbors if v is not ego and np.linalg.norm(v.position - ego.position) < 100]

        # Start state [x, y, vx, vy, psi]
        start_state = np.array([ego.position[0], ego.position[1], 
                                ego.velocity[0], 
                                ego.velocity[1], 
                                ego.heading]) # simplified velocity decomposition

        lane_center_ys = [l.position(0, 0)[1] for l in self.env.road.network.lanes_list()]

        # Initial guess: constant speed, zero steering
        initial_guess = np.zeros(self.horizon * 2) 
        
        # Bounds: limits on acc and steering
        bounds = []
        for _ in range(self.horizon):
            bounds.append((-self.acc_max, self.acc_max))
            bounds.append((-self.steering_max, self.steering_max))

        result = minimize(self.cost_function, 
                          initial_guess, 
                          args=(start_state, goal_phys, lane_center_ys, neighbors, steps_to_goal),
                          method='SLSQP',
                          bounds=bounds,
                          options={'ftol': 1e-3, 'disp': False})

        best_actions = result.x.reshape(-1, 2)
        
        # Calculate predicted trajectory to get next state
        optimal_traj = self.predict_trajectory(start_state, best_actions)
        pred_next_state = optimal_traj[1] # [x, y, vx, vy, psi]
        
        return best_actions[0], pred_next_state # Return first action [acc, steering] and predicted state

    def act(self, obs, goal_phys, steps_to_goal):
        action, pred_state = self.plan(obs, goal_phys, steps_to_goal)
        return action, pred_state
