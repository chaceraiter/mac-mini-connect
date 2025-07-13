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
    node_config: NodeConfig,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load only the layers assigned to this node, adapted for GPT-2 architecture.
    """
    model_name = MODEL_CONFIG["model_name"]
    device = device or MODEL_CONFIG["device"]
    dtype = getattr(torch, MODEL_CONFIG["dtype"])
    
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    
    ranges = get_layer_ranges(num_layers, node_config.world_size)
    my_start, my_end = ranges[node_config.rank]
    
    print(f"Node {node_config.name} loading layers {my_start} to {my_end}")
    
    partial_config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=my_end - my_start,
        torch_dtype=dtype,
    )
    
    model = AutoModelForCausalLM.from_config(partial_config)
    
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # Copy weights for our layers (using GPT-2's `transformer.h` structure)
    for i, layer_i in enumerate(range(my_start, my_end)):
        model.transformer.h[i].load_state_dict(
            full_model.transformer.h[layer_i].state_dict()
        )
    
    # First node gets token and position embeddings (`wte`, `wpe`)
    if node_config.rank == 0:
        model.transformer.wte = full_model.transformer.wte
        model.transformer.wpe = full_model.transformer.wpe

    # Last node gets the final layer norm and language model head
    if node_config.rank == node_config.world_size - 1:
        model.transformer.ln_f = full_model.transformer.ln_f
        model.lm_head = full_model.lm_head
    
    del full_model
    
    return model.to(device)

def forward_sequence(
    model: nn.Module,
    input_ids: torch.Tensor,
    node_config: NodeConfig
) -> torch.Tensor:
    """
    Forward pass handling the pipeline between nodes, adapted for GPT-2.
    """
    # First node: process embeddings
    if node_config.rank == 0:
        # Get token and positional embeddings
        token_embeddings = model.transformer.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        position_embeddings = model.transformer.wpe(position_ids)
        hidden_states = token_embeddings + position_embeddings
    else:
        hidden_states = input_ids
    
    # Process our layers (using GPT-2's `transformer.h` structure)
    for layer in model.transformer.h:
        layer_outputs = layer(
            hidden_states,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
    
    # Last node: process final layer and get logits
    if node_config.rank == node_config.world_size - 1:
        hidden_states = model.transformer.ln_f(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits
    
    return hidden_states 