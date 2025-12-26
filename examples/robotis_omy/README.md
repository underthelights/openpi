# Robotis OMY (OpenManipulator-X) OpenPI 예제

이 디렉토리는 OpenPI를 사용하여 Robotis OpenManipulator-X (OMY) 로봇을 제어하는 예제 코드와 환경 설정을 포함하고 있습니다.
Docker를 사용하여 **"로봇 제어부"**와 **"AI 추론부"**를 분리 또는 연동할 수 있도록 구성되어 있습니다.

## 1. 시스템 아키텍처 (System Architecture)

`./container.sh start` 명령을 실행하면 `compose.yml` 설정에 의해 **두 개의 컨테이너**가 준비됩니다. 
이 구성은 OpenManipulator의 낮은 사양 PC에서 추론 부하를 분리하기 위함입니다.

### (1) `robotis_omy` (이 예제의 핵심)
- **역할**: **"로봇의 몸체(Body)"**. 하드웨어 제어 및 드라이버 실행 전담.
- **내용물**: ROS 2 Jazzy, OpenManipulator 드라이버, RealSense 드라이버.
- **실행 위치**: 로봇과 USB로 연결된 PC (NUC, 노트북 등).
- **사용법**: 사용자는 **이 컨테이너 안에 들어가서** 드라이버를 켜고 클라이언트(`main.py`)를 실행해야 합니다.

### (2) `openpi_server` (선택 사항 / 참조용)
- **역할**: **"로봇의 두뇌(Brain)"**. 무거운 AI 모델(VLA)을 돌리는 서버.
- **내용물**: PyTorch, JAX, OpenPI 모델 가중치 등.
- **참고**: 만약 별도의 고성능 GPU 서버가 있다면 이 컨테이너는 무시해도 됩니다. 로봇이 연결된 PC에 GPU가 있다면 같이 띄워서 쓰면 됩니다.

---

## 2. 사전 준비 (Prerequisites)

### 하드웨어
- **로봇**: Robotis OpenManipulator-X (OM-X)
- **통신**: U2D2 또는 호환되는 USB-Serial 변환기
- **카메라**: Intel RealSense D435 등 (Realsense 드라이버 포함됨)

### 소프트웨어
- **Docker**: 로봇 제어 환경을 컨테이너로 실행하기 위해 필요합니다.

---

## 3. 설치 및 실행 (Docker)

### 컨테이너 시작 
먼저 컨테이너를 빌드하고 백그라운드에서 실행합니다.

```bash
cd examples/robotis_omy
./container.sh start
```

### 컨테이너 접속 (`robotis_omy`)
**로봇을 제어하려면 반드시 이 컨테이너로 들어가야 합니다.**

```bash
./container.sh enter
```

---

## 4. 로봇 및 센서 구동 (Inside `robotis_omy` Container)

컨테이너에 접속한 후(`enter`), 터미널을 여러 개 띄워(tmux 등) 아래 드라이버들을 각각 실행해주세요.

### (1) 로봇 드라이버 실행 (OpenManipulator-X)
로봇 팔 하드웨어와 통신을 시작합니다.
```bash
# 컨테이너 내부 터미널 1
ros2 launch open_manipulator_x_controller open_manipulator_x_controller.launch.py usb_port:=/dev/ttyUSB0
```

### (2) 카메라 실행 (RealSense)
카메라 이미지를 ROS 토픽으로 발행합니다.
```bash
# 컨테이너 내부 터미널 2
ros2 launch realsense2_camera rs_launch.py
```

### (3) 토픽 확인
모든 토픽이 제대로 나오는지 확인합니다.
```bash
ros2 topic list
# 필수 확인: /joint_states, /camera/..., /leader/arm_controller/joint_trajectory
```

---

## 5. OpenPI 클라이언트 실행 (Inside `robotis_omy` Container)

하드웨어가 준비되었으면, OpenPI 정책 서버(`openpi_server` 또는 외부 서버)에 접속하여 로봇을 움직입니다.

```bash
# 컨테이너 내부 터미널 3
# --host: AI 모델 서버의 IP (로컬이면 0.0.0.0 또는 localhost)
python3 examples/robotis_omy/main.py --host 0.0.0.0 --port 8000
```

---

## 6. 파일 구조 가이드

- `env.py`: **OpenPI Environment**. ROS2 데이터를 Gym 스타일(Observation)로 변환하고, Action을 받아 `bridge`로 넘깁니다.
- `ros2_bridge.py`: **ROS2 통신 담당**. 실제로 `rclpy`를 import하고 토픽을 구독/발행하는 노드입니다.
- `constants.py`: **설정 파일**. 토픽 이름이나 관절 이름이 다르면 여기를 수정하세요.
- `main.py`: **실행 파일**. 서버와 연결하고 로봇 Loop를 돌립니다.

## 7. 트러블슈팅

**Q. 로봇이 움직이지 않아요.**
A. `ros2 topic echo /joint_states`로 현재 관절 상태가 들어오는지, `ros2 topic echo /leader/arm_controller/joint_trajectory`로 명령이 발행되는지 확인하세요. 컨트롤러 이름이 맞는지 `constants.py`를 꼭 확인하세요.

**Q. 카메라 이미지가 안 보여요.**
A. `ros2 topic list`로 카메라 토픽명을 확인하고 `constants.py`의 `DEFAULT_CAMERA_TOPICS`를 실제 토픽명으로 수정해주세요.

**Q. `ModuleNotFoundError: No module named 'einops'` 에러**
A. 최신 코드로 빌드되지 않았을 수 있습니다. `docker compose up --build` 명령어로 이미지를 다시 빌드해보세요.