"""
Handles model sharding across multiple devices using pipeline parallelism.
This implementation is model-agnostic and driven by configuration.
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, List, Tuple
from .config import NodeConfig, MODEL_CONFIG
from .utils import get_nested_attr, set_nested_attr

def get_layer_ranges(total_layers: int, world_size: int) -> List[Tuple[int, int]]:
    """Calculate layer ranges for each device."""
    layers_per_device = total_layers // world_size
    ranges = []
    for i in range(world_size):
        start = i * layers_per_device
        end = total_layers if i == world_size - 1 else (i + 1) * layers_per_device
        ranges.append((start, end))
    return ranges

def load_partial_model(node_config: NodeConfig, device: Optional[str] = None) -> nn.Module:
    """
    Load only the layers and modules assigned to this node based on the
    model's architecture config.
    """
    model_name = MODEL_CONFIG["model_name"]
    arch_config = MODEL_CONFIG["model_arch_config"]
    device = device or MODEL_CONFIG["device"]
    dtype = getattr(torch, MODEL_CONFIG["dtype"])

    config = AutoConfig.from_pretrained(model_name)
    num_layers = get_nested_attr(config, arch_config["num_layers_path"])

    ranges = get_layer_ranges(num_layers, node_config.world_size)
    my_start, my_end = ranges[node_config.rank]
    
    print(f"Node {node_config.name} loading layers {my_start} to {my_end}")

    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    
    import copy
    model = copy.deepcopy(full_model)

    all_layers = get_nested_attr(full_model, arch_config["layers_path"])
    
    layers_to_keep = all_layers[my_start:my_end]
    set_nested_attr(model, arch_config["layers_path"], layers_to_keep)

    identity = nn.Identity()

    if node_config.rank != 0:
        set_nested_attr(model, arch_config["embedding_path"], identity)
        set_nested_attr(model, arch_config["positional_embedding_path"], identity)
        if arch_config.get("project_in_path"):
            set_nested_attr(model, arch_config["project_in_path"], identity)

    if node_config.rank != node_config.world_size - 1:
        set_nested_attr(model, arch_config["final_norm_path"], identity)
        set_nested_attr(model, arch_config["lm_head_path"], identity)
        if arch_config.get("project_out_path"):
            set_nested_attr(model, arch_config["project_out_path"], identity)

    del full_model, all_layers, layers_to_keep
    return model.to(device)

def forward_sequence(model: nn.Module, inputs: torch.Tensor, node_config: NodeConfig) -> torch.Tensor:
    """
    Generic forward pass that handles the pipeline between nodes based on model config.
    """
    arch_config = MODEL_CONFIG["model_arch_config"]
    hidden_states = inputs

    if node_config.rank == 0:
        token_embedder = get_nested_attr(model, arch_config["embedding_path"])
        pos_embedder = get_nested_attr(model, arch_config["positional_embedding_path"])
        
        token_embeddings = token_embedder(inputs)
        
        seq_length = inputs.size(1)
        pos_ids = torch.arange(seq_length, device=inputs.device).unsqueeze(0)
        
        offset = arch_config.get("positional_embedding_offset", 0)
        pos_embeddings = pos_embedder(pos_ids + offset)
        
        hidden_states = token_embeddings + pos_embeddings

        if arch_config.get("project_in_path"):
            project_in = get_nested_attr(model, arch_config["project_in_path"])
            if project_in and not isinstance(project_in, nn.Identity):
                hidden_states = project_in(hidden_states)

    layers = get_nested_attr(model, arch_config["layers_path"])
    for layer in layers:
        layer_outputs = layer(hidden_states, use_cache=False)
        hidden_states = layer_outputs[0]

    if node_config.rank == node_config.world_size - 1:
        if arch_config.get("project_out_path"):
            project_out = get_nested_attr(model, arch_config["project_out_path"])
            if project_out and not isinstance(project_out, nn.Identity):
                hidden_states = project_out(hidden_states)

        final_norm = get_nested_attr(model, arch_config["final_norm_path"])
        lm_head = get_nested_attr(model, arch_config["lm_head_path"])
        
        hidden_states = final_norm(hidden_states)
        logits = lm_head(hidden_states)
        return logits
    
    return hidden_states 