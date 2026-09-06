import numpy as np
from gymnasium.utils.env_checker import check_env

from mujoco_mobile_nav.envs.mobile_nav_env_0 import MobileNavEnv


def test_environment():
    env = MobileNavEnv("models/car.xml")

    check_env(env)

    env.close()


def test_reset():
    env = MobileNavEnv("models/car.xml")

    obs, info = env.reset(seed=42)

    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)

    env.close()


def test_step():
    env = MobileNavEnv("models/car.xml")

    obs, info = env.reset(seed=42)

    action = np.array([0.0, 0.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (5,)
    assert isinstance(reward, float | np.float32)
    assert isinstance(terminated, bool | np.bool_)
    assert isinstance(truncated, bool | np.bool_)
    assert isinstance(info, dict)

    env.close()