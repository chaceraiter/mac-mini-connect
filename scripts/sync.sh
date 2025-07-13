#!/bin/bash

# Define the remote project directory using a tilde for home directory
REMOTE_PROJECT_DIR="~/projects/mac-mini-connect/"

# Sync code to both Mac Minis
echo "Syncing to mini-red..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'venv' --exclude '.env' \
    ./ "mini-red@192.168.2.171:${REMOTE_PROJECT_DIR}"

echo "Syncing to mini-yellow..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'venv' --exclude '.env' \
    ./ "mini-yellow@192.168.2.224:${REMOTE_PROJECT_DIR}"

echo "Sync complete!" 