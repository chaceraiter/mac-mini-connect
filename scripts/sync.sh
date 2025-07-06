#!/bin/bash

# Sync code to both Mac Minis
echo "Syncing to mini-red..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'venv' --exclude '.env' \
    ./ "mini-red@mini-red:/Users/mini-red/projects/mac-mini-connect/"

echo "Syncing to mini-yellow..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'venv' --exclude '.env' \
    ./ "mini-yellow@mini-yellow:/Users/mini-yellow/projects/mac-mini-connect/"

echo "Sync complete!" 