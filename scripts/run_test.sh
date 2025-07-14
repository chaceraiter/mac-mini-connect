#!/bin/bash

# Kill any existing Python processes
echo "Cleaning up existing processes..."
ssh mini-red@192.168.2.171 "pkill -f python" || true
ssh mini-yellow@192.168.2.224 "pkill -f python" || true
sleep 2

# Define the project directory on the remote hosts
REMOTE_PROJECT_DIR="~/projects/mac-mini-connect"

# Set the model name
export MODEL_NAME="gpt2-xl"
export TEXT="Hello, my name is"

# Start master node (mini-red) first
echo "Starting master node (mini-red)..."
ssh mini-red@192.168.2.171 "export NODE_NAME=mini-red && export PYTORCH_ENABLE_MPS_FALLBACK=1 && cd ${REMOTE_PROJECT_DIR} && PYTHONPATH=${REMOTE_PROJECT_DIR} ${REMOTE_PROJECT_DIR}/venv/bin/python -m src.tests.test_sharding" 2>&1 | tee mini-red.log &
MASTER_PID=$!

# Give master node time to start up
echo "Waiting for master node to initialize..."
sleep 5

# Start worker node (mini-yellow)
echo "Starting worker node (mini-yellow)..."
ssh mini-yellow@192.168.2.224 "export NODE_NAME=mini-yellow && export PYTORCH_ENABLE_MPS_FALLBACK=1 && cd ${REMOTE_PROJECT_DIR} && PYTHONPATH=${REMOTE_PROJECT_DIR} ${REMOTE_PROJECT_DIR}/venv/bin/python -m src.tests.test_sharding" 2>&1 | tee mini-yellow.log &
WORKER_PID=$!

# Wait for both to finish
echo "Waiting for processes to complete..."
wait $MASTER_PID
wait $WORKER_PID 