import numpy as np
from mujoco_mobile_nav.controllers.simple_controller import simple_controller


def test_target_straight_ahead():
    obs = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    action = simple_controller(obs)

    np.testing.assert_allclose(action, [1.0, 0.0], atol=1e-6)


def test_target_to_left():
    obs = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    action = simple_controller(obs)

    assert action[0] == 1.0
    assert action[1] > 0.0


def test_target_to_right():
    obs = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    action = simple_controller(obs)

    assert action[0] == 1.0
    assert action[1] < 0.0


def test_target_behind():
    obs = np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32)

    action = simple_controller(obs)

    assert action[0] == 1.0
    # could turn left or right
    assert np.isclose(action[1], 1.0) | np.isclose(action[1], -1.0)
    
    
def test_manual_navigation():
    from mujoco_mobile_nav.envs.mobile_nav_env import MobileNavEnv

    env = MobileNavEnv("models/car.xml")

    obs, info = env.reset(seed=42)
    
    print(f"Initial state: {obs[:3]}")
    print(f"Target position: {obs[3:5]}")

    env.max_episode_steps = 5000
    for step in range(env.max_episode_steps):
        action = simple_controller(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        
    print(f"Final step: {step + 1}")
    print(f"Final observation: {obs}")
    print(f"Final distance to target: {info['distance_to_target']}")

    env.close()
    
    assert terminated


