"""Custom-config loading for the mp3d objnav scripts (Habitat v0.3.3 / Hydra).

Habitat v0.3.3 moved config from yacs (UPPERCASE, ``get_config(path)`` +
``defrost()``/``freeze()``) to Hydra/OmegaConf (lowercase, composable groups).

To keep our own experiment configs in a single ``configs/`` directory while
still composing the built-in habitat groups (``/benchmark/...``,
``/habitat/...``), we register a Hydra ``SearchPathPlugin`` that adds this
package's ``configs/`` directory as an extra provider (the vlfm / complex-objnav
approach).

Use ``load_config("objnav_mp3d.yaml", overrides=[...])`` instead of calling
``habitat.get_config`` directly. ``overrides`` is a list of dotlist strings,
e.g. ``["habitat.environment.max_episode_steps=1000",
"habitat.dataset.split=val"]`` -- this replaces the v0.2.2
``config.defrost(); config.X = ...; config.freeze()`` dance.
"""
import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from habitat.config.default import get_config
from habitat.config.default_structured_configs import register_hydra_plugin

# Absolute path to ./configs (cwd-independent).
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)


class MP3DObjNavConfigPlugin(SearchPathPlugin):
    """Adds ./configs to the Hydra search path (provider 'mp3d_objnav')."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="mp3d_objnav", path=CONFIG_DIR)


_REGISTERED = False


def _ensure_registered():
    global _REGISTERED
    if not _REGISTERED:
        register_hydra_plugin(MP3DObjNavConfigPlugin)
        _REGISTERED = True


def load_config(config_name, overrides=None):
    """Load one of our configs by name (relative to ./configs).

    Example: ``load_config("objnav_mp3d.yaml",
                            overrides=["habitat.dataset.split=val"])``.
    """
    _ensure_registered()
    return get_config(config_name, overrides=overrides, configs_dir=CONFIG_DIR)
