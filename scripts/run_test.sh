#!/bin/bash

# Kill any existing Python processes
echo "Cleaning up existing processes..."
ssh mini-red "pkill -f python" || true
ssh mini-yellow "pkill -f python" || true
sleep 2

# Start master node (mini-red) first
echo "Starting master node (mini-red)..."
ssh mini-red "cd /Users/mini-red/projects/mac-mini-connect && source venv/bin/activate && python src/test_sharding.py" 2>&1 | tee mini-red.log &
MASTER_PID=$!

# Give master node time to start up
echo "Waiting for master node to initialize..."
sleep 5

# Start worker node (mini-yellow)
echo "Starting worker node (mini-yellow)..."
ssh mini-yellow "cd /Users/mini-yellow/projects/mac-mini-connect && source venv/bin/activate && python src/test_sharding.py" 2>&1 | tee mini-yellow.log &
WORKER_PID=$!

# Wait for both to finish
echo "Waiting for processes to complete..."
wait $MASTER_PID
wait $WORKER_PID 