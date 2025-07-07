"""
Test script for model sharding across devices.
"""
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import sys

from common.config import get_node_config, MODEL_CONFIG
from common.distributed import setup_distributed, cleanup_distributed
from common.model_sharding import load_partial_model, forward_sequence

def log(msg):
    """Helper to ensure logs are flushed immediately"""
    print(f"[{dist.get_rank() if dist.is_initialized() else '?'}] {msg}", flush=True)

def main():
    # Initialize distributed setup
    node_config = get_node_config()
    log(f"Starting on node {node_config.name} (rank {node_config.rank})")
    log(f"Using master: {node_config.master_addr}:{node_config.master_port}")
    
    try:
        log("Setting up distributed...")
        setup_distributed(node_config)
        log("Distributed setup complete")
        
        # For testing, let's use a smaller model first
        model_name = "facebook/opt-350m"
        log(f"Loading tokenizer on {node_config.name}...")
        
        # Load tokenizer on all nodes (small enough to replicate)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log("Tokenizer loaded")
        
        log(f"Loading partial model on {node_config.name}...")
        # Load partial model on each node
        model = load_partial_model(model_name, node_config)
        log("Model loaded")
        
        # Test input
        text = "Hello, my name is"
        log(f"Processing text: {text}")
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(MODEL_CONFIG["device"])
        log(f"Input shape: {input_ids.shape}")
        
        # Process through pipeline
        hidden_states = input_ids
        
        # Forward pass through each node in sequence
        for rank in range(node_config.world_size):
            log(f"Processing rank {rank}")
            # Only the active rank processes
            if node_config.rank == rank:
                log(f"Node {node_config.name} processing...")
                hidden_states = forward_sequence(model, hidden_states, node_config)
                log(f"Node {node_config.name} processing complete")
                log(f"Output shape: {hidden_states.shape}")
            
            # Broadcast result to all nodes
            if rank < node_config.world_size - 1:
                log(f"Broadcasting from rank {rank}")
                hidden_states = hidden_states.contiguous()
                dist.broadcast(hidden_states, rank)
                log(f"Broadcast complete")
        
        # Only last node has the final logits
        if node_config.rank == node_config.world_size - 1:
            next_token = torch.argmax(hidden_states[0, -1])
            next_word = tokenizer.decode(next_token)
            log(f"Input: {text}")
            log(f"Next word prediction: {next_word}")
    
    except Exception as e:
        log(f"Error: {str(e)}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        log(f"Cleaning up...")
        cleanup_distributed()
        log(f"Cleanup complete")

if __name__ == "__main__":
    main() 