#!/bin/bash

# Configuration
REMOTE_USER_RED="mini-red"
REMOTE_HOST_RED="mini-red.lan"
REMOTE_IP_RED="192.168.2.171" # Using the primary IP, not the alias

REMOTE_USER_YELLOW="mini-yellow"
REMOTE_HOST_YELLOW="mini-yellow.lan"

REMOTE_PROJECT_DIR="/Users/$REMOTE_USER_RED/projects/mac-mini-connect" # Generic path structure

PORT=29502

# Function to clean up processes on a remote host
cleanup() {
    local user=$1
    local host=$2
    echo "Cleaning up processes on $host..."
    ssh "$user@$host" "pkill -f 'test_connection.py' || true"
}

# Clean up previous runs
cleanup $REMOTE_USER_RED $REMOTE_HOST_RED
cleanup $REMOTE_USER_YELLOW $REMOTE_HOST_YELLOW

# Define paths for red
VENV_PATH_RED="/Users/$REMOTE_USER_RED/projects/mac-mini-connect/venv/bin/activate"
SCRIPT_PATH_RED="/Users/$REMOTE_USER_RED/projects/mac-mini-connect/scripts/test_connection.py"

# Define paths for yellow
VENV_PATH_YELLOW="/Users/$REMOTE_USER_YELLOW/projects/mac-mini-connect/venv/bin/activate"
SCRIPT_PATH_YELLOW="/Users/$REMOTE_USER_YELLOW/projects/mac-mini-connect/scripts/test_connection.py"

# Start server on mini-red
echo "--- Starting server on $REMOTE_HOST_RED ---"
ssh "$REMOTE_USER_RED@$REMOTE_HOST_RED" "
    echo 'Setting up server environment...'
    source \"$VENV_PATH_RED\"
    echo 'Executing server script...'
    python \"$SCRIPT_PATH_RED\" server $REMOTE_IP_RED $PORT
" > server.log 2>&1 &
SERVER_PID=$!

# Wait a few seconds for the server to start
echo "Waiting for server to initialize..."
sleep 3

# Run client on mini-yellow
echo "--- Starting client on $REMOTE_HOST_YELLOW ---"
ssh "$REMOTE_USER_YELLOW@$REMOTE_HOST_YELLOW" "
    echo 'Setting up client environment...'
    source \"$VENV_PATH_YELLOW\"
    echo 'Executing client script...'
    python \"$SCRIPT_PATH_YELLOW\" client $REMOTE_IP_RED $PORT
" > client.log 2>&1
CLIENT_EXIT_CODE=$?

# Wait for server to finish
wait $SERVER_PID
SERVER_EXIT_CODE=$?

# --- Report Results ---
echo
echo "--- Server Log (mini-red) ---"
cat server.log

echo
echo "--- Client Log (mini-yellow) ---"
cat client.log

# Check results
if [ $SERVER_EXIT_CODE -eq 0 ] && [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo
    echo "✅ ✅ ✅ TEST SUCCEEDED ✅ ✅ ✅"
    exit 0
else
    echo
    echo "❌ ❌ ❌ TEST FAILED ❌ ❌ ❌"
    echo "Server exit code: $SERVER_EXIT_CODE"
    echo "Client exit code: $CLIENT_EXIT_CODE"
    exit 1
fi 