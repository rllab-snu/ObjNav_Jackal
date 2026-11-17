"""Small accessors that hide the v0.3.3 (Hydra/lowercase) config layout.

The v0.2.2 code read scalars off an UPPERCASE yacs config, e.g.
``config.SIMULATOR.RGB_SENSOR.WIDTH``. In v0.3.3 the same values live at
``config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width``.
These helpers centralize that mapping so the runners/agents stay readable and
there is exactly one place to touch if the layout shifts again.
"""
import numpy as np

# The agent key used in our configs (objectnav_mp3d benchmark uses "main_agent").
MAIN_AGENT = "main_agent"


def _agent_cfg(config):
    return config.habitat.simulator.agents[MAIN_AGENT]


def sim_sensor(config, name="rgb_sensor"):
    """Return a sim-sensor sub-config (rgb_sensor / depth_sensor / semantic_sensor)."""
    return _agent_cfg(config).sim_sensors[name]


def rgb_hw(config):
    """(width, height) of the RGB sensor."""
    s = sim_sensor(config, "rgb_sensor")
    return int(s.width), int(s.height)


def focal_length(config):
    """Pinhole focal length (px) derived from the RGB sensor width + HFOV."""
    s = sim_sensor(config, "rgb_sensor")
    return (int(s.width) / 2.0) / np.tan(np.deg2rad(float(s.hfov) / 2.0))


def max_episode_steps(config):
    return int(config.habitat.environment.max_episode_steps)


def forward_step_size(config):
    return float(config.habitat.simulator.forward_step_size)


def success_distance(config):
    return float(config.habitat.task.measurements.success.success_distance)


def scene_dataset(config):
    return config.habitat.simulator.get("scene_dataset", "default")


def dataset_type(config):
    return config.habitat.dataset.type


def dataset_split(config):
    return config.habitat.dataset.split
