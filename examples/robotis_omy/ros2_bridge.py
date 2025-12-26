# examples/robotis_omy/ros2_bridge.py
import threading
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import cv2

@dataclass
class Latest:
    qpos: np.ndarray | None = None
    images: dict | None = None

class OmyBridge(Node):
    def __init__(self, joint_names, joint_states_topic, camera_topics, traj_topic, traj_dt=0.2):
        super().__init__("openpi_omy_bridge")
        self._lock = threading.Lock()
        self._joint_names = list(joint_names)
        self._traj_dt = float(traj_dt)

        self._latest_qpos = None
        self._latest_images = {}

        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)

        for cam_name, topic in camera_topics.items():
            self.create_subscription(CompressedImage, topic, lambda msg, n=cam_name: self._on_image(n, msg), 10)

        self._traj_pub = self.create_publisher(JointTrajectory, traj_topic, 10)

    def _on_joint_state(self, msg: JointState):
        with self._lock:
            q = np.zeros((len(self._joint_names),), dtype=np.float32)
            for i, jn in enumerate(self._joint_names):
                if jn in msg.name:
                    q[i] = float(msg.position[msg.name.index(jn)])
            self._latest_qpos = q

    def _on_image(self, cam_name: str, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_images[cam_name] = rgb  # HWC uint8

    def get_qpos(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_qpos is None else self._latest_qpos.copy()

    def get_images(self) -> dict:
        with self._lock:
            return {k: v.copy() for k, v in self._latest_images.items()}

    def send_joint_trajectory(self, positions: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = self._joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions.tolist()]
        pt.time_from_start = Duration(sec=int(self._traj_dt), nanosec=int((self._traj_dt % 1.0) * 1e9))
        msg.points = [pt]

        self._traj_pub.publish(msg)
