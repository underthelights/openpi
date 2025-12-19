import dataclasses
import logging
import time
from typing import Dict, Any
import pathlib

import tyro
import tqdm
import yaml
import cv2

from openpi_client import websocket_client_policy as _websocket_client_policy
from env import make_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Args:
    # Server Host
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Policy Params
    num_steps: int = 1000  # Number of steps to run
    
    # OMY Configuration (can be loaded from yaml, but defaults here for simplicity)
    # This mirrors physical_ai_server config structure mostly
    joint_order_leader: tuple = (
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"
    )
    action_server_name: str = "/leader/joint_trajectory_controller/follow_joint_trajectory"

def main(args: Args):
    # 1. Setup Environment Config
    # We construct the params dict expected by OMYEnv
    env_params = {
        "action_server_name": args.action_server_name,
        "joint_order": {
            "leader": list(args.joint_order_leader)
        },
        "camera_topic_list": [
            "cam_wrist:/camera/cam_wrist/color/image_rect_raw/compressed",
            "cam_top:/camera/cam_top/color/image_rect_raw/compressed"
        ],
        "joint_topic_list": [
            "follower:/joint_states",
            "leader:/leader/joint_trajectory"
        ],
        "rosbag_extra_topic_list": [] # Required by Communicator init
    }
    
    # 2. Initialize Environment
    logger.info("Initializing OMY Environment...")
    env = make_env(env_params)
    
    # 3. Initialize Policy Client
    logger.info(f"Connecting to Policy Server at {args.host}:{args.port}...")
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        logger.info(f"Server metadata: {policy.get_server_metadata()}")
    except Exception as e:
        logger.error(f"Failed to connect to policy server: {e}")
        return

    # Start Viewer
    try:
        from viewer import start_viewer, update_frame
        start_viewer()
    except ImportError:
        logger.warning("Viewer module not found or flask not installed. Skipping viewer.")
        update_frame = None

    # 4. Inference Loop
    logger.info("Starting Inference Loop...")
    
    # Reset/Get first observation
    ts = env.reset()
    
    for _ in tqdm.trange(args.num_steps, desc="Running Policy"):
        if ts.observation is None:
             logger.warning("No observation received, skipping step.")
             time.sleep(0.1)
             continue

        # Update Viewer
        if update_frame:
            # images are in ts.observation['images'] dict
            # keys might be 'cam_high', 'cam_low', 'cam_wrist', etc.
            images = ts.observation.get("images", {})
            for name, img in images.items():
                # img is typically RGB (OpenPI convention)
                # cv2 expects BGR for encoding if input is RGB
                # But let's check format. OMYEnv output uses cv2.imdecode which gives BGR,
                # then OMYEnv converts BGR to RGB (Line 87 of omy_env.py).
                # So `img` here is RGB.
                # cv2.imencode in viewer.py expects BGR.
                # So we convert back to BGR for display.
                # Note: This double conversion is slightly inefficient but safe.
                try:
                    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    update_frame(name, bgr_img)
                except Exception as e:
                    pass

        # Infer action
        # OpenPI policies expect a dictionary observation
        # OMYEnv.get_observation returns {qpos, qvel, images} which is good.
        
        # We might need to adapt keys if the specific model expects different names
        # e.g. "state" instead of "qpos". 
        # For now, we pass as is or map if we know the model signature.
        # Assuming standard OpenPI model signature compatibility for now.
        
        action_dict = policy.infer(ts.observation)
        
        # Extract action for environment
        # The policy returns a dictionary, likely containing "actions" or similar.
        # We need to extract the joint positions.
        # If the model returns 'actions' as a numpy array:
        
        action = action_dict.get("actions")
        if action is None:
             # Fallback or check other keys
             action = action_dict.get("action")
        
        if action is not None:
             # Take step
             ts = env.step(action)
        else:
             logger.warning("Model did not return 'actions' or 'action' key.")

    # 5. Cleanup
    env.close()
    logger.info("Finished.")

if __name__ == "__main__":
    main(tyro.cli(Args))
