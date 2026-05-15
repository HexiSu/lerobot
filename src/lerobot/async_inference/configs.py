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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.common.hierarchical_task import (
    SUPPORTED_SUCCESS_DETECTORS,
    SUPPORTED_TASK_PLANNERS,
    parse_hsv,
    parse_roi,
)
from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Hierarchical task planning and success detection
    task_planner: str = field(
        default="none",
        metadata={"help": "Task planner to use. Options: none, rule_based, json, llm"},
    )
    task_plan_json: str | None = field(
        default=None, metadata={"help": "JSON task plan used when task_planner=json"}
    )
    subtask_timeout_s: float | None = field(
        default=None, metadata={"help": "Default timeout in seconds for each planned subtask"}
    )
    stop_when_task_plan_done: bool = field(
        default=False, metadata={"help": "Stop the client after the final subtask succeeds"}
    )
    clear_action_queue_on_subtask_change: bool = field(
        default=True, metadata={"help": "Discard queued actions when switching subtasks"}
    )
    success_detector: str = field(
        default="none",
        metadata={"help": "Subtask success detector. Options: none, timeout, orange_in_cup"},
    )
    success_camera: str = field(
        default="front", metadata={"help": "Camera key used by visual success detectors"}
    )
    cup_roi: str | None = field(
        default=None,
        metadata={"help": "Cup ROI as x1,y1,x2,y2 for success_detector=orange_in_cup"},
    )
    orange_hsv_lower: str = field(
        default="5,80,80", metadata={"help": "Lower HSV threshold for orange segmentation"}
    )
    orange_hsv_upper: str = field(
        default="25,255,255", metadata={"help": "Upper HSV threshold for orange segmentation"}
    )
    success_min_area: int = field(
        default=200, metadata={"help": "Minimum orange pixels inside cup ROI for success"}
    )
    success_hold_s: float = field(
        default=0.5, metadata={"help": "Seconds the success condition must hold before advancing"}
    )
    success_debug_view: bool = field(
        default=False, metadata={"help": "Show debug visualization for visual success detectors"}
    )

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.task_planner not in SUPPORTED_TASK_PLANNERS:
            available = sorted(SUPPORTED_TASK_PLANNERS)
            raise ValueError(f"task_planner must be one of {available}, got {self.task_planner}")

        if self.success_detector not in SUPPORTED_SUCCESS_DETECTORS:
            available = sorted(SUPPORTED_SUCCESS_DETECTORS)
            raise ValueError(f"success_detector must be one of {available}, got {self.success_detector}")

        if self.subtask_timeout_s is not None and self.subtask_timeout_s <= 0:
            raise ValueError(f"subtask_timeout_s must be positive, got {self.subtask_timeout_s}")

        if self.success_detector == "orange_in_cup" and self.cup_roi is None:
            raise ValueError("cup_roi is required when success_detector='orange_in_cup'")

        if self.cup_roi is not None:
            parse_roi(self.cup_roi)

        parse_hsv(self.orange_hsv_lower, "orange_hsv_lower")
        parse_hsv(self.orange_hsv_upper, "orange_hsv_upper")

        if self.success_min_area <= 0:
            raise ValueError(f"success_min_area must be positive, got {self.success_min_area}")

        if self.success_hold_s < 0:
            raise ValueError(f"success_hold_s must be non-negative, got {self.success_hold_s}")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "task_planner": self.task_planner,
            "task_plan_json": self.task_plan_json,
            "subtask_timeout_s": self.subtask_timeout_s,
            "stop_when_task_plan_done": self.stop_when_task_plan_done,
            "clear_action_queue_on_subtask_change": self.clear_action_queue_on_subtask_change,
            "success_detector": self.success_detector,
            "success_camera": self.success_camera,
            "cup_roi": self.cup_roi,
            "orange_hsv_lower": self.orange_hsv_lower,
            "orange_hsv_upper": self.orange_hsv_upper,
            "success_min_area": self.success_min_area,
            "success_hold_s": self.success_hold_s,
            "success_debug_view": self.success_debug_view,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
        }
