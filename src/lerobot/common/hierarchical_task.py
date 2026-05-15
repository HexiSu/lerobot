# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

RawObservation = dict[str, Any]

SUPPORTED_TASK_PLANNERS = {"none", "rule_based", "json", "llm"}
SUPPORTED_SUCCESS_DETECTORS = {"none", "timeout", "orange_in_cup"}


@dataclass(frozen=True)
class Subtask:
    instruction: str
    timeout_s: float | None = None
    success_condition: str | None = None


@dataclass(frozen=True)
class TaskPlan:
    original_task: str
    subtasks: list[Subtask]


@dataclass(frozen=True)
class Pi0LanguageInfo:
    pretrained_path: str | None
    paligemma_variant: str | None
    action_expert_variant: str | None
    tokenizer_name: str | None
    tokenizer_max_length: int | None


@dataclass(frozen=True)
class TaskTransition:
    previous_index: int
    previous_task: str
    current_index: int | None
    current_task: str | None
    reason: str
    done: bool


class TaskPlanner(Protocol):
    def plan(self, task: str) -> TaskPlan: ...


class SuccessDetector(Protocol):
    def is_success(self, subtask: Subtask, observation: RawObservation, elapsed_s: float) -> bool: ...

    def reset(self) -> None: ...


class RuleBasedTaskPlanner:
    _SEPARATORS = re.compile(
        r"\s*(?:,?\s*when\s+done\s*,?\s*then\s+|,?\s*after\s+that\s*,?\s*|\bthen\b|;|；|，?然后|，?接着)\s*",
        re.IGNORECASE,
    )

    def __init__(self, default_timeout_s: float | None = None):
        self.default_timeout_s = default_timeout_s

    def plan(self, task: str) -> TaskPlan:
        parts = [part.strip(" .。\t\n") for part in self._SEPARATORS.split(task) if part.strip(" .。\t\n")]
        if not parts:
            parts = [task.strip()]
        subtasks = [Subtask(instruction=part, timeout_s=self.default_timeout_s) for part in parts]
        return TaskPlan(original_task=task, subtasks=subtasks)


class JsonTaskPlanner:
    def __init__(self, task_plan_json: str):
        self.task_plan_json = task_plan_json

    def plan(self, task: str) -> TaskPlan:
        try:
            data = json.loads(self.task_plan_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid task_plan_json: {exc}") from exc

        subtask_items = data.get("subtasks", data) if isinstance(data, dict) else data
        if not isinstance(subtask_items, list):
            raise ValueError("task_plan_json must be a list or an object with a 'subtasks' list")

        subtasks = []
        for item in subtask_items:
            if isinstance(item, str):
                instruction = item.strip()
                timeout_s = None
                success_condition = None
            elif isinstance(item, dict):
                instruction = str(item.get("instruction", "")).strip()
                timeout_s = item.get("timeout_s")
                success_condition = item.get("success_condition")
            else:
                raise ValueError("Each subtask must be a string or object")

            if not instruction:
                raise ValueError("Subtask instruction cannot be empty")
            if timeout_s is not None:
                timeout_s = float(timeout_s)
                if timeout_s <= 0:
                    raise ValueError("Subtask timeout_s must be positive")
            subtasks.append(
                Subtask(instruction=instruction, timeout_s=timeout_s, success_condition=success_condition)
            )

        if not subtasks:
            raise ValueError("task_plan_json must contain at least one subtask")
        return TaskPlan(original_task=task, subtasks=subtasks)


class LLMTaskPlanner:
    def plan(self, task: str) -> TaskPlan:
        raise ValueError(
            "task_planner='llm' is not configured. The pi0 checkpoint language components are used for "
            "VLA action conditioning and are not exposed as a general task-planning LLM. Use "
            "task_planner='rule_based' or 'json', or add a configured LLM provider."
        )


class NoOpSuccessDetector:
    def is_success(self, subtask: Subtask, observation: RawObservation, elapsed_s: float) -> bool:
        return False

    def reset(self) -> None:
        pass


class TimeoutSuccessDetector:
    def is_success(self, subtask: Subtask, observation: RawObservation, elapsed_s: float) -> bool:
        return subtask.timeout_s is not None and elapsed_s >= subtask.timeout_s

    def reset(self) -> None:
        pass


class OrangeInCupDetector:
    def __init__(
        self,
        camera: str,
        cup_roi: tuple[int, int, int, int],
        orange_hsv_lower: tuple[int, int, int],
        orange_hsv_upper: tuple[int, int, int],
        min_area: int,
        hold_s: float,
        debug_view: bool = False,
        monotonic: Any = time.monotonic,
    ):
        self.camera = camera
        self.cup_roi = cup_roi
        self.orange_hsv_lower = np.array(orange_hsv_lower, dtype=np.uint8)
        self.orange_hsv_upper = np.array(orange_hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.hold_s = hold_s
        self.debug_view = debug_view
        self.monotonic = monotonic
        self._first_success_time: float | None = None

    def reset(self) -> None:
        self._first_success_time = None

    def is_success(self, subtask: Subtask, observation: RawObservation, elapsed_s: float) -> bool:
        if not self._applies_to_subtask(subtask):
            self._first_success_time = None
            return False

        image = observation.get(self.camera)
        if image is None:
            image = observation.get(f"observation.images.{self.camera}")
        if image is None:
            self._first_success_time = None
            return False

        image_np = np.asarray(image)
        if image_np.ndim != 3 or image_np.shape[2] < 3:
            self._first_success_time = None
            return False

        in_cup = self._orange_in_cup(image_np[..., :3])
        now = self.monotonic()
        if not in_cup:
            self._first_success_time = None
            return False
        if self._first_success_time is None:
            self._first_success_time = now
        return now - self._first_success_time >= self.hold_s

    def _applies_to_subtask(self, subtask: Subtask) -> bool:
        text = f"{subtask.instruction} {subtask.success_condition or ''}".lower()
        return any(keyword in text for keyword in ("orange", "cup", "橘", "杯"))

    def _orange_in_cup(self, image: np.ndarray) -> bool:
        import cv2

        x1, y1, x2, y2 = self.cup_roi
        height, width = image.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return False

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.orange_hsv_lower, self.orange_hsv_upper)
        roi_mask = mask[y1:y2, x1:x2]
        orange_area = int(np.count_nonzero(roi_mask))

        if self.debug_view:
            debug = image.copy()
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("orange_in_cup", cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
            cv2.imshow("orange_mask", mask)
            cv2.waitKey(1)

        return orange_area >= self.min_area


class HierarchicalTaskManager:
    def __init__(
        self,
        plan: TaskPlan,
        detector: SuccessDetector | None = None,
        monotonic: Any = time.monotonic,
    ):
        if not plan.subtasks:
            raise ValueError("Task plan must contain at least one subtask")
        self.plan = plan
        self.detector = detector or NoOpSuccessDetector()
        self.monotonic = monotonic
        self.current_index = 0
        self._subtask_started_at = self.monotonic()
        self._done = False
        self._pending_transition: TaskTransition | None = None

    @property
    def enabled(self) -> bool:
        return len(self.plan.subtasks) > 1 or not isinstance(self.detector, NoOpSuccessDetector)

    def current_task(self) -> str:
        if self._done:
            return self.plan.subtasks[-1].instruction
        return self.plan.subtasks[self.current_index].instruction

    def update(self, observation: RawObservation | None = None) -> None:
        if self._done or observation is None:
            return
        subtask = self.plan.subtasks[self.current_index]
        elapsed_s = self.monotonic() - self._subtask_started_at
        if self.detector.is_success(subtask, observation, elapsed_s):
            self.advance("success_detector")

    def mark_success(self, reason: str = "manual_success") -> None:
        self.advance(reason)

    def advance(self, reason: str) -> None:
        if self._done:
            return

        previous_index = self.current_index
        previous_task = self.plan.subtasks[previous_index].instruction
        next_index = previous_index + 1

        if next_index >= len(self.plan.subtasks):
            self._done = True
            self._pending_transition = TaskTransition(
                previous_index=previous_index,
                previous_task=previous_task,
                current_index=None,
                current_task=None,
                reason=reason,
                done=True,
            )
            return

        self.current_index = next_index
        self._subtask_started_at = self.monotonic()
        self.detector.reset()
        self._pending_transition = TaskTransition(
            previous_index=previous_index,
            previous_task=previous_task,
            current_index=self.current_index,
            current_task=self.plan.subtasks[self.current_index].instruction,
            reason=reason,
            done=False,
        )

    def is_done(self) -> bool:
        return self._done

    def consume_transition(self) -> TaskTransition | None:
        transition = self._pending_transition
        self._pending_transition = None
        return transition


def parse_int_tuple(value: str, expected_len: int, name: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated integers")
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must contain only integers") from exc
    return parsed


def parse_roi(value: str) -> tuple[int, int, int, int]:
    roi = parse_int_tuple(value, 4, "cup_roi")
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        raise ValueError("cup_roi must satisfy x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def parse_hsv(value: str, name: str) -> tuple[int, int, int]:
    hsv = parse_int_tuple(value, 3, name)
    if any(channel < 0 or channel > 255 for channel in hsv):
        raise ValueError(f"{name} values must be between 0 and 255")
    return hsv[0], hsv[1], hsv[2]


def load_pi0_language_info(pretrained_name_or_path: str) -> Pi0LanguageInfo | None:
    path = Path(pretrained_name_or_path)
    try:
        if not path.exists():
            return None

        config_path = path / "config.json"
        preprocessor_path = path / "policy_preprocessor.json"
        if not config_path.exists():
            return None

        config = json.loads(config_path.read_text())
    except OSError:
        return None
    tokenizer_name = None
    tokenizer_max_length = config.get("tokenizer_max_length")

    if preprocessor_path.exists():
        preprocessor = json.loads(preprocessor_path.read_text())
        for step in preprocessor.get("steps", []):
            step_config = step.get("config", {})
            if step.get("registry_name") == "tokenizer_processor":
                tokenizer_name = step_config.get("tokenizer_name")
                tokenizer_max_length = step_config.get("max_length", tokenizer_max_length)
                break

    return Pi0LanguageInfo(
        pretrained_path=config.get("pretrained_path"),
        paligemma_variant=config.get("paligemma_variant"),
        action_expert_variant=config.get("action_expert_variant"),
        tokenizer_name=tokenizer_name,
        tokenizer_max_length=tokenizer_max_length,
    )


def make_task_planner(
    task_planner: str,
    task_plan_json: str | None = None,
    default_timeout_s: float | None = None,
) -> TaskPlanner | None:
    if task_planner == "none":
        return None
    if task_planner == "rule_based":
        return RuleBasedTaskPlanner(default_timeout_s=default_timeout_s)
    if task_planner == "json":
        if not task_plan_json:
            raise ValueError("task_plan_json is required when task_planner='json'")
        return JsonTaskPlanner(task_plan_json)
    if task_planner == "llm":
        return LLMTaskPlanner()
    raise ValueError(f"Unsupported task_planner: {task_planner}")


def make_success_detector(
    success_detector: str,
    *,
    camera: str,
    cup_roi: str | None,
    orange_hsv_lower: str,
    orange_hsv_upper: str,
    min_area: int,
    hold_s: float,
    debug_view: bool,
) -> SuccessDetector:
    if success_detector == "none":
        return NoOpSuccessDetector()
    if success_detector == "timeout":
        return TimeoutSuccessDetector()
    if success_detector == "orange_in_cup":
        if cup_roi is None:
            raise ValueError("cup_roi is required when success_detector='orange_in_cup'")
        if min_area <= 0:
            raise ValueError("success_min_area must be positive")
        if hold_s < 0:
            raise ValueError("success_hold_s must be non-negative")
        return OrangeInCupDetector(
            camera=camera,
            cup_roi=parse_roi(cup_roi),
            orange_hsv_lower=parse_hsv(orange_hsv_lower, "orange_hsv_lower"),
            orange_hsv_upper=parse_hsv(orange_hsv_upper, "orange_hsv_upper"),
            min_area=min_area,
            hold_s=hold_s,
            debug_view=debug_view,
        )
    raise ValueError(f"Unsupported success_detector: {success_detector}")


def make_task_manager(
    task: str,
    *,
    task_planner: str,
    task_plan_json: str | None,
    default_timeout_s: float | None,
    detector: SuccessDetector,
) -> HierarchicalTaskManager | None:
    planner = make_task_planner(task_planner, task_plan_json, default_timeout_s)
    if planner is None:
        return None
    plan = planner.plan(task)
    return HierarchicalTaskManager(plan=plan, detector=detector)
