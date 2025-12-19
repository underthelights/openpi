import logging
import threading
import time
from typing import Any, Dict

import dm_env
import rclpy
from rclpy.executors import MultiThreadedExecutor

# Assuming physical_ai_tools is in PYTHONPATH
from physical_ai_tools.external_adapter.omy_env import OMYEnv

logger = logging.getLogger(__name__)

class OMYOpenPIEnv:
    """
    OpenPI-compatible wrapper for OMYEnv.
    Handles ROS 2 node lifecycle and background spinning.
    """
    def __init__(self, params: Dict[str, Any]):
        # Initialize ROS 2 context if not already active
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("openpi_omy_client")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        # Start executor in a background thread
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        # Initialize OMYEnv
        self.env = OMYEnv(self.node, params)

    def reset(self) -> dm_env.TimeStep:
        # OMYEnv doesn't have a reset in the viewed code, 
        # but usually we just want the latest observation for VLA
        # or we might need to move to a home position.
        # For now, we return the current observation.
        obs = self.env.get_observation()
        # If obs is None, we might want to wait or retry, handled in get_observation usually.
        
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(action)

    def close(self):
        self.env.cleanup()
        self.executor.shutdown()
        self.node.destroy_node()
        # Check if we should shutdown rclpy (only if we started it?)
        # rclpy.shutdown() 
        # Usually better to leave global shutdown to the main script or atexit

def make_env(config: Dict[str, Any]) -> OMYOpenPIEnv:
    return OMYOpenPIEnv(config)
