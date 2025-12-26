# examples/robotis_omy/constants.py

DEFAULT_CAMERA_TOPICS = {
    "cam_wrist_color": "/camera/cam_wrist/color/image_rect_raw/compressed",
    "cam_third_color": "/camera/cam_third/color/image_raw/compressed",
    "cam_top": "/camera/cam_top/color/image_raw/compressed",
}

DEFAULT_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"
]

# 너희 시스템이 실제로 쓰는 trajectory 토픽으로 맞추기
DEFAULT_TRAJECTORY_TOPIC = "/leader/arm_controller/joint_trajectory"  # or "/arm_controller/joint_trajectory"
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"

RENDER_H, RENDER_W = 224, 224

# 그리퍼 범위(OMY F3M 텔레오프 코드에서 max/min이 이렇게 잡혀있음)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.1
