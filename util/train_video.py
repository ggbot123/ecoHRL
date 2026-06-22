from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo


VIDEO_ENV_CHOICES = {"env0", "all"}


class FixedObserverRender(gym.Wrapper):
    """Render with a fixed observer position instead of following the ego vehicle."""

    class _Observer:
        def __init__(self, position):
            self.position = np.asarray(position, dtype=float)

    def __init__(self, env: gym.Env, position):
        super().__init__(env)
        self._observer = self._Observer(position)

    def _install_observer(self) -> None:
        raw_env = self.env.unwrapped
        if getattr(raw_env, "viewer", None) is None:
            # Let the environment initialize its own viewer, then replace the
            # observer with a fixed global-camera dummy.
            self.env.render()
        raw_env.viewer.observer_vehicle = self._observer

    def render(self):
        self._install_observer()
        return self.env.render()


def global_view_video_config() -> dict:
    return {
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": True,
        "warmup_render": False,
        "offscreen_rendering": True,
    }


def validate_video_env_choice(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in VIDEO_ENV_CHOICES:
        raise ValueError(f"{name} must be 'env0' or 'all'")
    return normalized


def env_index_set(choice: str, n_envs: int) -> set[int]:
    return set(range(n_envs)) if choice == "all" else {0}


def make_episode_trigger(video_episode_freq: int, scheduled: bool):
    if not bool(scheduled):
        return lambda ep: False
    freq = max(1, int(video_episode_freq))
    return lambda ep: (int(ep) % freq) == 0


def wrap_training_video_env(
    env: gym.Env,
    video_folder: str | None,
    video_name_prefix: str,
    video_episode_freq: int,
    record_video_scheduled: bool,
) -> gym.Env:
    if not video_folder:
        raise ValueError("video_folder must be provided when record_video=True")

    raw_env = env.unwrapped
    center_x = float(getattr(raw_env, "config", {}).get("road_length", 0.0)) / 2.0
    env = FixedObserverRender(env, position=[center_x, 5.0])

    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=make_episode_trigger(video_episode_freq, record_video_scheduled),
        name_prefix=video_name_prefix,
        disable_logger=True,
    )
    try:
        env.unwrapped.set_record_video_wrapper(env)
    except Exception:
        pass
    return env
