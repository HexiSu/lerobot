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

from pathlib import Path

import numpy as np
import pytest

from lerobot.common.hierarchical_task import (
    HierarchicalTaskManager,
    JsonTaskPlanner,
    OrangeInCupDetector,
    RuleBasedTaskPlanner,
    Subtask,
    TaskPlan,
    TimeoutSuccessDetector,
    load_pi0_language_info,
)


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rule_based_planner_splits_when_done_then():
    plan = RuleBasedTaskPlanner().plan("Grab Orange, when done, then make a face")

    assert [subtask.instruction for subtask in plan.subtasks] == ["Grab Orange", "make a face"]


def test_rule_based_planner_keeps_single_task():
    plan = RuleBasedTaskPlanner().plan("Grab Orange")

    assert [subtask.instruction for subtask in plan.subtasks] == ["Grab Orange"]


def test_rule_based_planner_splits_chinese_connector():
    plan = RuleBasedTaskPlanner().plan("抓橘子，然后做鬼脸")

    assert [subtask.instruction for subtask in plan.subtasks] == ["抓橘子", "做鬼脸"]


def test_json_task_planner_accepts_list_of_objects():
    plan = JsonTaskPlanner(
        '[{"instruction": "Grab Orange", "timeout_s": 2}, {"instruction": "make a face"}]'
    ).plan("ignored")

    assert plan.subtasks == [
        Subtask(instruction="Grab Orange", timeout_s=2.0, success_condition=None),
        Subtask(instruction="make a face", timeout_s=None, success_condition=None),
    ]


def test_json_task_planner_rejects_invalid_json():
    with pytest.raises(ValueError, match="Invalid task_plan_json"):
        JsonTaskPlanner("not json").plan("ignored")


def test_record_dataset_config_validates_hierarchical_options():
    from lerobot.scripts.lerobot_record import DatasetRecordConfig

    with pytest.raises(ValueError, match="cup_roi is required"):
        DatasetRecordConfig(
            repo_id="test/repo",
            single_task="Grab Orange, when done, then make a face",
            task_planner="rule_based",
            success_detector="orange_in_cup",
        )

    cfg = DatasetRecordConfig(
        repo_id="test/repo",
        single_task="Grab Orange, when done, then make a face",
        task_planner="rule_based",
        success_detector="orange_in_cup",
        cup_roi="10,20,30,40",
    )

    assert cfg.task_planner == "rule_based"
    assert cfg.success_detector == "orange_in_cup"


def test_load_pi0_language_info_from_local_checkpoint():
    checkpoint = Path("outputs/train/pi0/checkpoints/last/pretrained_model")
    if not checkpoint.exists():
        pytest.skip("Local pi0 checkpoint is not available")

    info = load_pi0_language_info(str(checkpoint))

    assert info is not None
    assert info.tokenizer_name == "google/paligemma-3b-pt-224"
    assert info.paligemma_variant == "gemma_2b"
    assert info.action_expert_variant == "gemma_300m"


def test_task_manager_advances_on_timeout_detector():
    clock = FakeClock()
    plan = TaskPlan(
        original_task="demo",
        subtasks=[Subtask("Grab Orange", timeout_s=1.0), Subtask("make a face", timeout_s=1.0)],
    )
    manager = HierarchicalTaskManager(plan, detector=TimeoutSuccessDetector(), monotonic=clock)

    clock.advance(1.1)
    manager.update({})

    assert manager.current_task() == "make a face"
    transition = manager.consume_transition()
    assert transition is not None
    assert transition.reason == "success_detector"
    assert not transition.done


def test_task_manager_done_after_final_success():
    plan = TaskPlan(original_task="demo", subtasks=[Subtask("Grab Orange")])
    manager = HierarchicalTaskManager(plan)

    manager.mark_success()

    assert manager.is_done()
    transition = manager.consume_transition()
    assert transition is not None
    assert transition.done


def test_orange_in_cup_detector_success_inside_roi():
    pytest.importorskip("cv2")
    clock = FakeClock()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[40:60, 40:60] = np.array([255, 140, 0], dtype=np.uint8)
    detector = OrangeInCupDetector(
        camera="front",
        cup_roi=(30, 30, 70, 70),
        orange_hsv_lower=(5, 80, 80),
        orange_hsv_upper=(25, 255, 255),
        min_area=20,
        hold_s=0.5,
        monotonic=clock,
    )

    assert not detector.is_success(Subtask("Grab Orange"), {"front": image}, elapsed_s=0.0)
    clock.advance(0.6)

    assert detector.is_success(Subtask("Grab Orange"), {"front": image}, elapsed_s=0.6)


def test_orange_in_cup_detector_ignores_unrelated_subtask():
    pytest.importorskip("cv2")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[40:60, 40:60] = np.array([255, 140, 0], dtype=np.uint8)
    detector = OrangeInCupDetector(
        camera="front",
        cup_roi=(30, 30, 70, 70),
        orange_hsv_lower=(5, 80, 80),
        orange_hsv_upper=(25, 255, 255),
        min_area=20,
        hold_s=0.0,
    )

    assert not detector.is_success(Subtask("make a face"), {"front": image}, elapsed_s=0.0)


def test_orange_in_cup_detector_false_outside_roi():
    pytest.importorskip("cv2")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:30, 10:30] = np.array([255, 140, 0], dtype=np.uint8)
    detector = OrangeInCupDetector(
        camera="front",
        cup_roi=(40, 40, 80, 80),
        orange_hsv_lower=(5, 80, 80),
        orange_hsv_upper=(25, 255, 255),
        min_area=20,
        hold_s=0.0,
    )

    assert not detector.is_success(Subtask("Grab Orange"), {"front": image}, elapsed_s=0.0)
