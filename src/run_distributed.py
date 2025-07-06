"""
Distributed ML testing script that can run on any node.
"""
import os
import socket
import torch
from common import NODES, setup_distributed_env, cleanup_distributed, is_master, synchronize

def get_node_config():
    """Determine which node this is based on hostname."""
    hostname = socket.gethostname()
    for node_config in NODES.values():
        if node_config.name in hostname:
            return node_config
    raise ValueError(f"Unknown host: {hostname}. Must be one of: {list(NODES.keys())}")

def main():
    # Figure out which node we are
    node_config = get_node_config()
    print(f"Starting up on {node_config.name} (rank {node_config.rank})")
    
    # Set up distributed environment
    setup_distributed_env(node_config)
    
    try:
        # Both nodes do the same work, just with different ranks
        if is_master():
            print("I am the master node (rank 0)")
        else:
            print("I am a worker node")
            
        # Synchronize to make sure all nodes are ready
        synchronize()
        print(f"Node {node_config.name} ready for distributed operations")
        
        # TODO: Add actual ML work here
        
    finally:
        # Clean up
        cleanup_distributed()

if __name__ == "__main__":
    main() 