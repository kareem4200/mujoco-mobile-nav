import numpy as np


def simple_controller(obs):
    
    x, y, theta, target_x, target_y = obs
    
    dx = target_x - x
    dy = target_y - y
    
    heading_error = np.arctan2(dy, dx) - theta
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # wrap to [-pi, pi]
    
    # simple proportional controller for steering
    forward_speed = 1.0
    angular_speed = 2.0 * heading_error
    
    angular_speed = np.clip(angular_speed, -1.0, 1.0)
    
    return np.array([forward_speed, angular_speed], dtype=np.float32)