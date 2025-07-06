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
    )
}

# Model configurations
MODEL_CONFIG = {
    "model_name": "facebook/opt-350m",  # Starting with a smaller model for testing
    "device": "mps",  # Metal Performance Shaders for Apple Silicon
    "dtype": "float16",  # Mixed precision for better performance
}

def get_node_config() -> NodeConfig:
    """
    Get the configuration for the current node based on hostname.
    Returns:
        NodeConfig: Configuration for this node
    Raises:
        ValueError: If hostname doesn't match any configured node
    """
    hostname = socket.gethostname().split('.')[0]  # Get short hostname
    
    # Try exact match first
    if hostname in NODES:
        return NODES[hostname]
    
    # Try matching by prefix (in case hostname has additional suffixes)
    for name, config in NODES.items():
        if hostname.startswith(name):
            return config
    
    raise ValueError(f"No configuration found for hostname: {hostname}") 