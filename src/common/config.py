"""
Configuration settings for distributed ML testing.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import socket

@dataclass
class NodeConfig:
    """Configuration for a single node in the distributed setup."""
    name: str
    address: str
    rank: int
    world_size: int = 2  # Total number of nodes
    master_addr: str = "0.0.0.0"  # Bind to all interfaces
    master_port: int = 29501  # Using a different port to avoid conflicts
    backend: str = "gloo"  # Using gloo as NCCL isn't available on macOS

# Pre-configured nodes
NODES: Dict[str, NodeConfig] = {
    "mini-red": NodeConfig(
        name="mini-red",
        address="192.168.2.171",
        rank=0,  # Master node
        master_addr="192.168.2.171",  # Other nodes need the actual IP to connect
    ),
    "mini-yellow": NodeConfig(
        name="mini-yellow",
        address="192.168.2.224",
        rank=1,  # Worker node
        master_addr="192.168.2.171", # Must point to the master node
    )
}

# Model configurations
MODEL_CONFIG = {
    "model_name": "gpt2",  # Standard model, no projection layer.
    "device": "mps",  # Metal Performance Shaders for Apple Silicon
    "dtype": "float16",  # Mixed precision for better performance
}

def get_node_config() -> NodeConfig:
    """
    Get the configuration for the current node based on the NODE_NAME env var.
    Returns:
        NodeConfig: Configuration for this node
    Raises:
        ValueError: If NODE_NAME is not set or invalid
    """
    import os
    node_name = os.environ.get("NODE_NAME")
    if not node_name:
        raise ValueError("The NODE_NAME environment variable must be set (e.g., 'mini-red', 'mini-yellow').")
        
    if node_name in NODES:
        return NODES[node_name]
    
    raise ValueError(f"No configuration found for node name: {node_name}") 