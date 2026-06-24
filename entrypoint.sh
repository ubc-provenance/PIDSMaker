#!/bin/bash
set -e

CONTAINER_USER="pids"
CONTAINER_HOME="/home/user"

# Detect the host user's UID/GID from the bind-mounted project directory.
# Bind mounts preserve the host file ownership, so stat gives us the host UID/GID
# without needing any environment variables.
TARGET_UID=$(stat /home/pids -c '%u' 2>/dev/null || echo "1000")
TARGET_GID=$(stat /home/pids -c '%g' 2>/dev/null || echo "1000")

# Safety guard: never remap to root
[ "$TARGET_UID" = "0" ] && TARGET_UID=1000
[ "$TARGET_GID" = "0" ] && TARGET_GID=1000

CURRENT_UID=$(id -u "$CONTAINER_USER")
CURRENT_GID=$(id -g "$CONTAINER_USER")

if [ "$TARGET_UID" != "$CURRENT_UID" ] || [ "$TARGET_GID" != "$CURRENT_GID" ]; then
    groupmod -g "$TARGET_GID" "$CONTAINER_USER" 2>/dev/null || true
    usermod -u "$TARGET_UID" -g "$TARGET_GID" "$CONTAINER_USER" 2>/dev/null || true
    # Re-own the user's home directory (not the bind-mounted project dir)
    chown -R "$TARGET_UID:$TARGET_GID" "$CONTAINER_HOME" 2>/dev/null || true
fi

# Ensure the artifacts directory exists and is writable by the container user.
# This handles both the case where Docker created it as root and the case
# where it doesn't exist yet on the host.
mkdir -p /home/artifacts
chown "$TARGET_UID:$TARGET_GID" /home/artifacts

exec gosu "$CONTAINER_USER" "$@"
