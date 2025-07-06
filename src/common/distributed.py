"""
Utilities for distributed ML setup and communication.
"""
import os
import torch
import torch.distributed as dist
from typing import Optional
from .config import NodeConfig

def setup_distributed(node_config: NodeConfig) -> None:
    """
    Set up the distributed environment for a node.
    
    Args:
        node_config: Configuration for this node
    """
    os.environ['MASTER_ADDR'] = node_config.master_addr
    os.environ['MASTER_PORT'] = str(node_config.master_port)
    os.environ['WORLD_SIZE'] = str(node_config.world_size)
    os.environ['RANK'] = str(node_config.rank)
    
    # Initialize the distributed process group
    dist.init_process_group(
        backend=node_config.backend,
        init_method=f"tcp://{node_config.master_addr}:{node_config.master_port}",
        world_size=node_config.world_size,
        rank=node_config.rank
    )

def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_master() -> bool:
    """Check if this is the master node."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_local_rank() -> int:
    """Get the local rank of this node."""
    if not dist.is_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))

def synchronize() -> None:
    """Synchronize all processes."""
    if not dist.is_initialized():
        return
    dist.barrier() 