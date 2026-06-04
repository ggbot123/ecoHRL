from __future__ import annotations

import gc
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


class CollisionAwareRecordVideo(RecordVideo):
    """Record scheduled videos and always keep episodes that contain collisions.

    Gymnasium's RecordVideo decides whether to record at episode start, before we
    know whether the episode will crash. To avoid missing collision episodes, this
    wrapper records every episode when collision capture is enabled, then deletes
    unscheduled non-collision videos after the episode finishes.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger,
        name_prefix: str,
        keep_collision_episodes: bool,
        disable_logger: bool = True,
    ):
        self._regular_episode_trigger = episode_trigger
        self._keep_collision_episodes = bool(keep_collision_episodes)
        self._episode_had_collision = False
        self._active_regular_recording = False
        self._active_video_name = None
        self.recorded_frames = []
        super().__init__(
            env,
            video_folder=video_folder,
            episode_trigger=self._recording_episode_trigger,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )

    def _recording_episode_trigger(self, episode_id: int) -> bool:
        regular = bool(self._regular_episode_trigger(int(episode_id)))
        self._active_regular_recording = regular
        return regular or self._keep_collision_episodes

    @staticmethod
    def _info_has_collision(info: dict) -> bool:
        if not isinstance(info, dict):
            return False
        if bool(info.get("crashed", False)):
            return True
        rewards = info.get("rewards", None)
        if isinstance(rewards, dict):
            try:
                return float(rewards.get("collision_reward", 0.0)) > 0.0
            except (TypeError, ValueError):
                return False
        return False

    def _raw_env_has_collision(self) -> bool:
        vehicle = getattr(self.unwrapped, "vehicle", None)
        return bool(getattr(vehicle, "crashed", False))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._episode_had_collision = False
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        if self._info_has_collision(info) or self._raw_env_has_collision():
            self._episode_had_collision = True
        return obs, rew, terminated, truncated, info

    def start_recording(self, video_name: str):
        self._active_video_name = str(video_name)
        return super().start_recording(video_name)

    def stop_recording(self):
        keep_video = bool(self._active_regular_recording or self._episode_had_collision)
        assert self.recording, "stop_recording was called, but no recording was started"

        if keep_video and len(self.recorded_frames) > 0:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            try:
                clip.write_videofile(path, logger=moviepy_logger)
            finally:
                close = getattr(clip, "close", None)
                if callable(close):
                    close()
                del clip

        self.recorded_frames = []
        self.recording = False
        self._video_name = None
        self._active_video_name = None
        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()


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
    record_video_global_view: bool,
    record_video_scheduled: bool,
    record_video_collision_episodes: bool,
) -> gym.Env:
    if not video_folder:
        raise ValueError("video_folder must be provided when record_video=True")

    if record_video_global_view:
        raw_env = env.unwrapped
        center_x = float(getattr(raw_env, "config", {}).get("road_length", 0.0)) / 2.0
        env = FixedObserverRender(env, position=[center_x, 5.0])

    os.makedirs(video_folder, exist_ok=True)
    env = CollisionAwareRecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=make_episode_trigger(video_episode_freq, record_video_scheduled),
        name_prefix=video_name_prefix,
        keep_collision_episodes=bool(record_video_collision_episodes),
        disable_logger=True,
    )
    try:
        env.unwrapped.set_record_video_wrapper(env)
    except Exception:
        pass
    return env
