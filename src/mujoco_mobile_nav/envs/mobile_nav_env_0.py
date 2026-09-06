import gymnasium as gym
from gymnasium import spaces
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

class MobileNavEnv(gym.Env):

    def __init__(self, model_path):
        # MuJoCo model
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # x, y, z, qw, qx, qy, qz, left_wheel_angle, right_wheel_angle
        self.initial_state = np.array([0.0, 
                                       0.0, 
                                       0.03, 
                                       1.0, 
                                       0.0, 
                                       0.0, 
                                       0.0,
                                       0.0, 
                                       0.0], dtype=np.float32)
        self.initial_target = np.array([3.0, 5.0], dtype=np.float32)
        self.target_threshold = 0.1
        
        # action space (linear velocity, angular velocity)
        # self.max_linear_velocity = 1.0  # m/s
        # self.max_angular_velocity = 0.5  # rad/s
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # observation space (x, y, theta, target_x, target_y)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.max_episode_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset step counter
        self.current_step = 0

        # Reset robot
        mj.mj_resetData(self.model, self.data)
        
        # randomize initial state
        self.data.qpos[:2] = self.np_random.uniform(-1.0, 1.0, size=2) 
        random_theta = self.np_random.uniform(-np.pi, np.pi)
        theta_quat = Rotation.from_euler('z', random_theta).as_quat(scalar_first=True)
        self.data.qpos[3:7] = theta_quat
        self.data.qvel[:] = 0.0
        
        # randomize target position
        self.initial_target = np.array([self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)], dtype=np.float32)
        
        mj.mj_forward(self.model, self.data)
        
        # return observation
        # rot = np.zeros(9)
        # mj.mju_quat2Mat(rot, self.data.qpos[3:7])
        # theta = np.arctan2(rot[3], rot[0])
        
        theta = Rotation.from_quat(self.data.qpos[3:7]).as_euler('xyz')[2]
        
        obs = np.array([self.data.qpos[0], 
                        self.data.qpos[1], 
                        theta, 
                        self.initial_target[0], 
                        self.initial_target[1]], dtype=np.float32)
        
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mj.mj_step(self.model, self.data)
        self.current_step += 1

        # rot = np.zeros(9)
        # mj.mju_quat2Mat(rot, self.data.qpos[3:7])
        # theta = np.arctan2(rot[3], rot[0])
        theta = Rotation.from_quat(self.data.qpos[3:7], scalar_first=True).as_euler('xyz')[2]
        
        obs = np.array([self.data.qpos[0], 
                        self.data.qpos[1], 
                        theta, 
                        self.initial_target[0], 
                        self.initial_target[1]], dtype=np.float32)
        
        distance_to_target = np.linalg.norm(obs[:2] - obs[3:5])
        reward = -distance_to_target
        terminated = distance_to_target < self.target_threshold
        truncated = self.current_step >= self.max_episode_steps
        info = {
            "distance_to_target": distance_to_target,
            "current_step": self.current_step,
            "is_success": terminated
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
    
if __name__ == "__main__":
    env = MobileNavEnv("models/car.xml")

    obs, info = env.reset()
    
    print("Initial Observation:", obs)
    
    for _ in range(1000):
        # action = env.action_space.sample()
        action = np.array([1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Action:", action)
        print("Observation:", obs, "Reward:", reward)
        
        if terminated or truncated:
            break