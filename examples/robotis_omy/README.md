# Robotis OMY OpenPI Example

This example demonstrates how to control a Robotis OMY (OpenManipulator-X) robot using the OpenPI framework.

## Prerequisites

- **Hardware**: Robotis OpenManipulator-X, Intel RealSense Config (U2D2 etc).
- **Driver**: This example includes a Dockerfile that builds the necessary ROS 2 Jazzy environment and drivers.
- **Physical AI Tools**: The `physical_ai_tools` repository is required.

## Configuration

The `compose.yml` is configured for:
- **ROS 2 Distribution**: Jazzy
- **Network Mode**: Host (Required for ROS communication)
- **Privileges**: High (Access to `/dev`, `sys_nice` capability)
- **Domain ID**: 30 (Default)

## Usage

### Using Helper Script (Recommended)

This directory includes a `container.sh` script to manage the lifecycle easily.

1.  **Start the Container**:
    ```bash
    ./examples/robotis_omy/container.sh start
    ```
    This builds the image and starts the container in the background. It also sets up necessary udev rules.

2.  **Enter the Container**:
    ```bash
    ./examples/robotis_omy/container.sh enter
    ```
    This opens a bash shell inside the running container.

3.  **Stop the Container**:
    ```bash
    ./examples/robotis_omy/container.sh stop
    ```

### Manual Usage
You can still use docker compose directly:
```bash
cd examples/robotis_omy
docker compose up --build
```

## Customization
- **Policy Server**: Ensure your policy server is running at the host defined in `main.py` (default `0.0.0.0:8000`).
- **Action Server**: If you need to change the controller name, edit `main.py` or pass `--action-server-name`.

## Notes
- This container runs in `privileged` mode with `host` networking to ensure low-latency hardware control.