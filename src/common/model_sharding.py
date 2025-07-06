"""
Handles model sharding across multiple devices using pipeline parallelism.
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, List, Tuple
from .config import NodeConfig, MODEL_CONFIG

def get_layer_ranges(total_layers: int, world_size: int) -> List[Tuple[int, int]]:
    """
    Calculate layer ranges for each device.
    Returns list of (start_layer, end_layer) tuples.
    """
    layers_per_device = total_layers // world_size
    ranges = []
    for i in range(world_size):
        start = i * layers_per_device
        # For last device, include any remaining layers
        end = total_layers if i == world_size - 1 else (i + 1) * layers_per_device
        ranges.append((start, end))
    return ranges

def load_partial_model(
    model_name: str,
    node_config: NodeConfig,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load only the layers assigned to this node.
    
    Args:
        model_name: HuggingFace model name
        node_config: Configuration for this node
        device: Device to place model on (default: from MODEL_CONFIG)
    """
    device = device or MODEL_CONFIG["device"]
    dtype = getattr(torch, MODEL_CONFIG["dtype"])
    
    # First get model config to know number of layers
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    
    # Calculate which layers belong to this node
    ranges = get_layer_ranges(num_layers, node_config.world_size)
    my_start, my_end = ranges[node_config.rank]
    
    print(f"Node {node_config.name} loading layers {my_start} to {my_end}")
    
    # Load full model config but modify for partial layers
    partial_config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=my_end - my_start,
        torch_dtype=dtype,
    )
    
    # Initialize a partial model with modified config
    model = AutoModelForCausalLM.from_config(partial_config)
    
    # Load pretrained weights for our layers only
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # Copy weights for our layers
    for i, layer_i in enumerate(range(my_start, my_end)):
        model.model.decoder.layers[i].load_state_dict(
            full_model.model.decoder.layers[layer_i].state_dict()
        )
    
    # Copy embeddings only to first node and output layer only to last node
    if node_config.rank == 0:
        model.model.decoder.embed_tokens = full_model.model.decoder.embed_tokens
        model.model.decoder.embed_positions = full_model.model.decoder.embed_positions
    if node_config.rank == node_config.world_size - 1:
        model.model.decoder.final_layer_norm = full_model.model.decoder.final_layer_norm
        model.lm_head = full_model.lm_head
    
    del full_model  # Free up memory
    
    return model.to(device)

def forward_sequence(
    model: nn.Module,
    input_ids: torch.Tensor,
    node_config: NodeConfig
) -> torch.Tensor:
    """
    Forward pass handling the pipeline between nodes.
    
    Args:
        model: The partial model on this node
        input_ids: Input token IDs
        node_config: Configuration for this node
    """
    # First node: process embeddings
    if node_config.rank == 0:
        # Create attention mask (1 for all tokens since we're not doing padding)
        attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        hidden_states = model.model.decoder.embed_tokens(input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden_states = hidden_states + model.model.decoder.embed_positions(position_ids)
    else:
        hidden_states = input_ids
    
    # Process our layers
    for layer in model.model.decoder.layers:
        layer_outputs = layer(
            hidden_states,
            attention_mask=None,  # Not needed for OPT in this simple case
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False
        )
        hidden_states = layer_outputs[0]
    
    # Last node: process final layers and get logits
    if node_config.rank == node_config.world_size - 1:
        hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits
    
    return hidden_states 