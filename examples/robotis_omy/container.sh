#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_NAME="openpi_omy_robot" # Must match container_name in compose.yml

# Function to display help
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  help                    Show this help message"
    echo "  start                   Start the container (builds if needed)"
    echo "  enter                   Enter the running container"
    echo "  stop                    Stop the container"
    echo ""
    echo "Examples:"
    echo "  $0 start                Start container"
    echo "  $0 enter                Enter the running container"
    echo "  $0 stop                 Stop the container"
}

# Function to start the container
start_container() {
    # Set up X11 forwarding only if DISPLAY is set
    if [ -n "$DISPLAY" ]; then
        echo "Setting up X11 forwarding..."
        xhost +local:docker || true
    else
        echo "Warning: DISPLAY environment variable is not set. X11 forwarding will not be available."
    fi

    echo "Starting Robotis OMY OpenPI container..."

    # Copy udev rule for FTDI (U2D2) - requires sudo
    # Note: If user doesn't have sudo passwordless, this might prompt.
    echo 'KERNEL=="ttyUSB*", DRIVERS=="ftdi_sio", MODE="0666", ATTR{device/latency_timer}="1"' | sudo tee /etc/udev/rules.d/99-u2d2.rules > /dev/null

    # Reload udev rules
    echo "Reloading udev rules..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger

    # Build and Run
    # Using compose.yml from the same directory
    docker compose -f "${SCRIPT_DIR}/compose.yml" up -d --build
}

# Function to enter the container
enter_container() {
    # Set up X11 forwarding only if DISPLAY is set
    if [ -n "$DISPLAY" ]; then
        xhost +local:docker || true
    fi

    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        echo "Error: Container '$CONTAINER_NAME' is not running."
        echo "Try running '$0 start' first."
        exit 1
    fi
    docker exec -it "$CONTAINER_NAME" bash
}

# Function to stop the container
stop_container() {
    echo "Stopping container..."
    docker compose -f "${SCRIPT_DIR}/compose.yml" down
}

# Main command handling
case "$1" in
    "help")
        show_help
        ;;
    "start")
        start_container
        ;;
    "enter")
        enter_container
        ;;
    "stop")
        stop_container
        ;;
    *)
        # Default to showing help if no valid command
        if [ -z "$1" ]; then
            show_help
        else
            echo "Error: Unknown command '$1'"
            show_help
            exit 1
        fi
        ;;
esac
