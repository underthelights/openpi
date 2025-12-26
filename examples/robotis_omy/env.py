# examples/robotis_omy/env.py
import threading
from typing import Optional, List

import einops
import numpy as np
from typing_extensions import override

import rclpy
from rclpy.executors import MultiThreadedExecutor

from openpi_client import image_tools
from openpi_client.runtime import environment as _environment

from examples.robotis_omy import constants
from examples.robotis_omy.ros2_bridge import OmyBridge


class OmyRealEnvironment(_environment.Environment):
    def __init__(
        self,
        reset_position: Optional[List[float]] = None,
        render_height: int = constants.RENDER_H,
        render_width: int = constants.RENDER_W,
        joint_names: Optional[List[str]] = None,
        camera_topics: Optional[dict] = None,
        traj_topic: str = constants.DEFAULT_TRAJECTORY_TOPIC,
        joint_states_topic: str = constants.DEFAULT_JOINT_STATES_TOPIC,
        traj_dt: float = 0.2,
        gripper_min: float = constants.GRIPPER_MIN,
        gripper_max: float = constants.GRIPPER_MAX,
    ):
        self._render_height = render_height
        self._render_width = render_width
        self._reset_position = np.array(reset_position, dtype=np.float32) if reset_position is not None else None
        self._gripper_min = float(gripper_min)
        self._gripper_max = float(gripper_max)

        if not rclpy.ok():
            rclpy.init(args=None)

        self._bridge = OmyBridge(
            joint_names=joint_names or constants.DEFAULT_JOINT_NAMES,
            joint_states_topic=joint_states_topic,
            camera_topics=camera_topics or constants.DEFAULT_CAMERA_TOPICS,
            traj_topic=traj_topic,
            traj_dt=traj_dt,
        )

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._bridge)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    @override
    def reset(self) -> None:
        # (선택) reset_position이 있으면 그 포즈로 이동
        if self._reset_position is not None:
            self._bridge.send_joint_trajectory(self._reset_position)

    @override
    def is_episode_complete(self) -> bool:
        return False  # 필요하면 버튼/시간/성공조건으로 바꾸기

    @override
    def get_observation(self) -> dict:
        qpos = self._bridge.get_qpos()
        if qpos is None:
            # joint_states 아직 못 받았으면 0으로 (혹은 block)
            qpos = np.zeros((7,), dtype=np.float32)

        images = self._bridge.get_images()

        # openpi 예제처럼 CHW uint8로 통일【openpi.txt†L500-L526】
        out_images = {}
        for cam_name, img_hwc in images.items():
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_hwc, self._render_height, self._render_width)
            )
            out_images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {"state": qpos.astype(np.float32), "images": out_images}

    @override
    def apply_action(self, action: dict) -> None:
        a = np.asarray(action["actions"], dtype=np.float32)

        # gripper clamp (OMY 범위)
        if a.shape[0] >= 7:
            a[6] = np.clip(a[6], self._gripper_min, self._gripper_max)

        self._bridge.send_joint_trajectory(a)
