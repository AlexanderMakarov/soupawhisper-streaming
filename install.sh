#!/bin/bash
# Install SoupaWhisper on Linux and macOS
# Linux: systemd user service (Ubuntu, Pop!_OS, Debian, Fedora, Arch)
# macOS: launchd user LaunchAgent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HOME/.config/soupawhisper"
SERVICE_DIR="$HOME/.config/systemd/user"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_LABEL="com.soupawhisper.dictate"
LOG_FILE="$HOME/Library/Logs/soupawhisper.log"

is_macos() {
    [ "$(uname -s)" = "Darwin" ]
}

# Get venv path (used by both Linux and macOS service install)
get_venv_path() {
    local pm
    pm=$(detect_python_manager)
    case $pm in
        uv)
            echo "$SCRIPT_DIR/.venv"
            ;;
        poetry)
            poetry env info --path 2>/dev/null || echo "$SCRIPT_DIR/.venv"
            ;;
        *)
            echo "$SCRIPT_DIR/.venv"
            ;;
    esac
}

# Detect package manager
detect_package_manager() {
    if command -v apt &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    elif command -v zypper &> /dev/null; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Install system dependencies (Linux only)
install_deps() {
    if is_macos; then
        echo "macOS detected: skipping system package install (no extra deps required)"
        return 0
    fi
    local pm=$(detect_package_manager)

    echo "Detected package manager: $pm"
    echo "Installing system dependencies..."

    case $pm in
        apt)
            sudo apt update
            sudo apt install -y alsa-utils xclip xdotool libnotify-bin
            ;;
        dnf)
            sudo dnf install -y alsa-utils xclip xdotool libnotify
            ;;
        pacman)
            sudo pacman -S --noconfirm alsa-utils xclip xdotool libnotify
            ;;
        zypper)
            sudo zypper install -y alsa-utils xclip xdotool libnotify-tools
            ;;
        *)
            echo "Unknown package manager. Please install manually:"
            echo "  alsa-utils xclip xdotool libnotify"
            ;;
    esac
}

# Detect Python package manager
detect_python_manager() {
    if command -v uv &> /dev/null; then
        echo "uv"
    elif command -v poetry &> /dev/null; then
        echo "poetry"
    else
        echo "unknown"
    fi
}

# Install Python dependencies
install_python() {
    echo ""
    echo "Installing Python dependencies..."

    local pm=$(detect_python_manager)

    case $pm in
        uv)
            echo "Using uv..."
            uv sync
            ;;
        poetry)
            echo "Using Poetry..."
            poetry install
            ;;
        *)
            echo "Neither uv nor Poetry found. Please install one:"
            echo "  uv:     curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  Poetry: curl -sSL https://install.python-poetry.org | python3 -"
            exit 1
            ;;
    esac
}

# Setup config file
setup_config() {
    echo ""
    echo "Setting up config..."
    mkdir -p "$CONFIG_DIR"

    if [ ! -f "$CONFIG_DIR/config.ini" ]; then
        cp "$SCRIPT_DIR/config.example.ini" "$CONFIG_DIR/config.ini"
        echo "Created config at $CONFIG_DIR/config.ini"
    else
        echo "Config already exists at $CONFIG_DIR/config.ini"
    fi
}

# Install systemd service (Linux)
install_service_linux() {
    echo ""
    echo "Installing systemd user service..."

    mkdir -p "$SERVICE_DIR"

    # Get current display settings
    local display="${DISPLAY:-:0}"
    local xauthority="${XAUTHORITY:-$HOME/.Xauthority}"
    local venv_path
    venv_path=$(get_venv_path)

    cat > "$SERVICE_DIR/soupawhisper.service" << EOF
[Unit]
Description=SoupaWhisper Voice Dictation
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
ExecStart=$venv_path/bin/python $SCRIPT_DIR/dictate.py
Restart=on-failure
RestartSec=5

# X11 display access
Environment=DISPLAY=$display
Environment=XAUTHORITY=$xauthority

[Install]
WantedBy=default.target
EOF

    echo "Created service at $SERVICE_DIR/soupawhisper.service"

    # Reload and enable
    systemctl --user daemon-reload
    systemctl --user enable soupawhisper

    # Restart if running, otherwise start
    if systemctl --user is-active --quiet soupawhisper 2>/dev/null; then
        echo "Restarting service..."
        systemctl --user restart soupawhisper
    else
        echo "Starting service..."
        systemctl --user start soupawhisper || echo "Service start failed (may need manual start)"
    fi

    echo ""
    echo "Service installed! Commands:"
    echo "  make service-start   # or systemctl --user start soupawhisper"
    echo "  make service-stop    # or systemctl --user stop soupawhisper"
    echo "  make service-status  # or systemctl --user status soupawhisper"
    echo "  make service-logs    # or journalctl --user -u soupawhisper -f"
}

# Install launchd service (macOS)
install_service_macos() {
    echo ""
    echo "Installing launchd user service (LaunchAgent)..."

    mkdir -p "$LAUNCH_AGENTS"
    mkdir -p "$(dirname "$LOG_FILE")"

    local venv_path
    venv_path=$(get_venv_path)
    local python_bin="$venv_path/bin/python"
    if [ ! -x "$python_bin" ]; then
        echo "Error: Python not found at $python_bin. Run install without --skip-deps first."
        exit 1
    fi

    cat > "$LAUNCH_AGENTS/$PLIST_LABEL.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$python_bin</string>
        <string>$SCRIPT_DIR/dictate.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

    echo "Created plist at $LAUNCH_AGENTS/$PLIST_LABEL.plist"
    echo "Log file: $LOG_FILE"

    # Unload if already loaded (e.g. reinstall)
    launchctl unload "$LAUNCH_AGENTS/$PLIST_LABEL.plist" 2>/dev/null || true
    launchctl load "$LAUNCH_AGENTS/$PLIST_LABEL.plist"

    echo ""
    echo "Service installed! Commands:"
    echo "  make service-start   # or launchctl load $LAUNCH_AGENTS/$PLIST_LABEL.plist"
    echo "  make service-stop    # or launchctl unload $LAUNCH_AGENTS/$PLIST_LABEL.plist"
    echo "  make service-status  # or launchctl list | grep $PLIST_LABEL"
    echo "  make service-logs    # tail -f $LOG_FILE"
}

install_service() {
    if is_macos; then
        install_service_macos
    else
        install_service_linux
    fi
}

# Main
main() {
    local SKIP_DEPS=false
    local INSTALL_SERVICE=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps|--no-deps)
                SKIP_DEPS=true
                shift
                ;;
            --install-service|--service)
                INSTALL_SERVICE=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--skip-deps] [--install-service]"
                echo ""
                echo "Options:"
                echo "  --skip-deps, --no-deps        Skip installation of OS dependencies (requires sudo)"
                echo "  --install-service, --service   Install systemd (Linux) or launchd (macOS) service without prompting"
                echo "  -h, --help                    Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done

    echo "==================================="
    echo "  SoupaWhisper Installer"
    echo "==================================="
    echo ""

    if [ "$SKIP_DEPS" = false ]; then
        install_deps
    else
        echo "Skipping OS dependencies installation (--skip-deps flag provided)"
    fi

    install_python
    setup_config

    if [ "$INSTALL_SERVICE" = true ]; then
        install_service
    else
        echo ""
        if is_macos; then
            read -p "Install as launchd service (LaunchAgent)? [y/N] " -n 1 -r
        else
            read -p "Install as systemd service? [y/N] " -n 1 -r
        fi
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_service
        fi
    fi

    echo ""
    echo "==================================="
    echo "  Installation complete!"
    echo "==================================="
    echo ""
    echo "To run manually:"
    local pm=$(detect_python_manager)
    case $pm in
        uv)
            echo "  uv run python dictate.py"
            ;;
        poetry)
            echo "  poetry run python dictate.py"
            ;;
        *)
            echo "  python dictate.py  # (after activating venv)"
            ;;
    esac
    echo ""
    echo "Config: $CONFIG_DIR/config.ini"
    hotkey_key=$(grep -E '^[[:space:]]*key[[:space:]]*=' "$CONFIG_DIR/config.ini" 2>/dev/null | sed -n 's/^[[:space:]]*key[[:space:]]*=[[:space:]]*\(.*\)/\1/p' | tr -d ' ' | tail -1)
    if [ -n "$hotkey_key" ]; then
        hotkey_display=$(echo "$hotkey_key" | tr '[:lower:]' '[:upper:]')
    else
        hotkey_display="F12"
    fi
    echo "Hotkey: $hotkey_display (hold to record)"
    echo "Exit:   Ctrl+C"
}

main "$@"
