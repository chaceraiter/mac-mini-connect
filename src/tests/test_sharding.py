"""
Test script for model sharding across devices.
"""
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers import AutoConfig

from src.common.config import get_node_config, MODEL_CONFIG
from src.common.distributed import setup_distributed, cleanup_distributed
from src.common.model_sharding import load_partial_model, forward_sequence
from src.common.utils import get_nested_attr

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
        
        # Use the model defined in the central config
        model_name = MODEL_CONFIG["model_name"]
        log(f"Loading tokenizer for {model_name} on {node_config.name}...")
        
        # Load tokenizer on all nodes (small enough to replicate)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log("Tokenizer loaded")
        
        log(f"Loading partial model on {node_config.name}...")
        # Load partial model on each node
        model = load_partial_model(node_config)
        log("Model loaded")
        
        # Test input
        text = "Hello, my name is"
        log(f"Processing text: {text}")
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(MODEL_CONFIG["device"])
        log(f"Input shape: {input_ids.shape}")
        
        # --- Pipeline Execution ---
        if node_config.rank == 0:
            # Rank 0 starts with token IDs and gets intermediate hidden states
            log("Rank 0: Executing forward pass...")
            hidden_states = forward_sequence(model, input_ids, node_config)
            log(f"Rank 0: Output shape: {hidden_states.shape}")
            
            # Send hidden states to the next node
            log("Rank 0: Broadcasting hidden states to Rank 1...")
            dist.broadcast(hidden_states, src=0)
            log("Rank 0: Broadcast complete.")

        elif node_config.rank == 1:
            # Rank 1 needs a placeholder to receive the hidden states
            config = AutoConfig.from_pretrained(model_name)
            hidden_size = get_nested_attr(config, MODEL_CONFIG["model_arch_config"]["hidden_size_path"])
            # Shape: [batch_size, sequence_length, hidden_size]
            placeholder_shape = (input_ids.shape[0], input_ids.shape[1], hidden_size)
            hidden_states = torch.empty(
                placeholder_shape,
                dtype=getattr(torch, MODEL_CONFIG["dtype"]),
                device=MODEL_CONFIG["device"]
            )
            
            # Receive hidden states from the previous node
            log("Rank 1: Waiting to receive hidden states from Rank 0...")
            dist.broadcast(hidden_states, src=0)
            log("Rank 1: Received hidden states.")

            # Rank 1 processes the hidden states to get final logits
            log("Rank 1: Executing forward pass...")
            logits = forward_sequence(model, hidden_states, node_config)
            log(f"Rank 1: Output logits shape: {logits.shape}")
            
            # Decode the result
            next_token = torch.argmax(logits[0, -1])
            next_word = tokenizer.decode(next_token)
            log(f"Input: '{text}'")
            log(f"Next word prediction: '{next_word}'")
    
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