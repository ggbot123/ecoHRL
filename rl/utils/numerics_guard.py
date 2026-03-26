from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

import numpy as np
import torch as th


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _arr_to_str(x: Any) -> str:
    arr = _to_numpy(x)
    return np.array2string(arr, precision=6, separator=",", suppress_small=False, threshold=10_000)


def _bad_rows(t: th.Tensor) -> np.ndarray:
    """Return batch indices that contain non-finite values.

    For scalar tensors, returns array([0]) when non-finite, else empty array.
    """
    mask = ~th.isfinite(t)
    if mask.ndim == 0:
        return np.array([0], dtype=np.int64) if bool(mask.item()) else np.array([], dtype=np.int64)
    if mask.ndim == 1:
        return th.nonzero(mask, as_tuple=False).reshape(-1).detach().cpu().numpy().astype(np.int64)
    reduce_dims = tuple(range(1, mask.ndim))
    row_bad = mask.any(dim=reduce_dims)
    return th.nonzero(row_bad, as_tuple=False).reshape(-1).detach().cpu().numpy().astype(np.int64)


@dataclass
class SACNumericsGuardConfig:
    enabled: bool = False
    save_dir: str = "./logs"
    file_name: str = "sac_non_finite_debug.csv"
    max_rows_per_event: int = 8


class SACNumericsGuard:
    """Detect non-finite values during SAC updates, dump forensic CSV, and stop training."""

    def __init__(self, cfg: SACNumericsGuardConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.csv_path = os.path.join(cfg.save_dir, cfg.file_name)
        self._header_written = False
        self._event_id = 0

    @classmethod
    def from_dict(cls, cfg_dict: Optional[dict[str, Any]]) -> "SACNumericsGuard":
        if cfg_dict is None:
            return cls(SACNumericsGuardConfig())
        cfg = SACNumericsGuardConfig(
            enabled=bool(cfg_dict.get("enabled", False)),
            save_dir=str(cfg_dict.get("save_dir", "./logs")),
            file_name=str(cfg_dict.get("file_name", "sac_non_finite_debug.csv")),
            max_rows_per_event=int(cfg_dict.get("max_rows_per_event", 8)),
        )
        return cls(cfg)

    def _ensure_csv(self) -> None:
        if self._header_written:
            return
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "event_id",
                        "algo_update",
                        "gradient_step",
                        "bad_items",
                        "bad_index",
                        "obs",
                        "action",
                        "reward",
                        "next_obs",
                        "done",
                        "target_q",
                        "next_action",
                        "target_q_values_all",
                        "current_q_values_all",
                        "next_log_prob",
                        "critic_loss",
                    ]
                )
        self._header_written = True

    def _append_rows(self, rows: Iterable[list[Any]]) -> None:
        self._ensure_csv()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def check_and_raise(
        self,
        *,
        algo_update: int,
        gradient_step: int,
        replay_data: Any,
        next_actions: th.Tensor,
        next_log_prob: th.Tensor,
        target_q_values_all: th.Tensor,
        target_q_values: th.Tensor,
        current_q_values_all: th.Tensor,
        critic_loss: th.Tensor,
    ) -> None:
        if not self.enabled:
            return

        check_map: dict[str, th.Tensor] = {
            "obs": replay_data.observations,
            "action": replay_data.actions,
            "reward": replay_data.rewards,
            "next_obs": replay_data.next_observations,
            "done": replay_data.dones,
            "next_action": next_actions,
            "next_log_prob": next_log_prob,
            "target_q_values_all": target_q_values_all,
            "target_q": target_q_values,
            "current_q_values_all": current_q_values_all,
            "critic_loss": critic_loss,
        }

        bad_items: list[str] = []
        bad_row_set: set[int] = set()
        for name, tensor in check_map.items():
            rows = _bad_rows(tensor)
            if rows.size > 0:
                bad_items.append(name)
                bad_row_set.update(int(i) for i in rows.tolist())

        if not bad_items:
            return

        self._event_id += 1
        bad_rows = sorted(list(bad_row_set))
        if len(bad_rows) == 0:
            bad_rows = [0]
        bad_rows = bad_rows[: max(1, int(self.cfg.max_rows_per_event))]

        ts = datetime.now().isoformat(timespec="seconds")
        rows_to_write: list[list[Any]] = []
        for bi in bad_rows:
            rows_to_write.append(
                [
                    ts,
                    self._event_id,
                    int(algo_update),
                    int(gradient_step),
                    "|".join(bad_items),
                    int(bi),
                    _arr_to_str(replay_data.observations[bi]),
                    _arr_to_str(replay_data.actions[bi]),
                    _arr_to_str(replay_data.rewards[bi]),
                    _arr_to_str(replay_data.next_observations[bi]),
                    _arr_to_str(replay_data.dones[bi]),
                    _arr_to_str(target_q_values[bi]),
                    _arr_to_str(next_actions[bi]),
                    _arr_to_str(target_q_values_all[bi]),
                    _arr_to_str(current_q_values_all[bi]),
                    _arr_to_str(next_log_prob[bi]),
                    _arr_to_str(critic_loss),
                ]
            )
        self._append_rows(rows_to_write)

        print(
            "[SACNumericsGuard] Detected non-finite values. "
            f"items={bad_items}, update={algo_update}, grad_step={gradient_step}, "
            f"saved_csv={self.csv_path}"
        )
        raise RuntimeError(
            "SACNumericsGuard aborted training due to non-finite values in: " + ", ".join(bad_items)
        )
