"""
Handles model sharding across multiple devices using pipeline parallelism.
This implementation is model-agnostic and driven by configuration.
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, List, Tuple
import datetime
import os
import psutil
from .config import NodeConfig, MODEL_CONFIG
from .utils import get_nested_attr, set_nested_attr

def _log(rank, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"[{timestamp}] [{rank}] {message}", flush=True)

def check_cache(model_name: str) -> bool:
    """Checks if a model is likely cached by Hugging Face."""
    # This is a heuristic, not a guaranteed check.
    # It checks for the presence of the model's snapshot directory.
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        from huggingface_hub.utils import hub_folder_name
        
        model_cache_path = os.path.join(HUGGINGFACE_HUB_CACHE, f"models--{hub_folder_name(repo_id=model_name)}")
        return os.path.exists(model_cache_path)
    except Exception as e:
        return False # Fail safe

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
    num_layers = get_nested_attr(config, arch_config["num_layers_key"])

    ranges = get_layer_ranges(num_layers, node_config.world_size)
    my_start, my_end = ranges[node_config.rank]
    
    _log(node_config.rank, f"Node {node_config.name} loading layers {my_start} to {my_end}")

    is_cached = check_cache(model_name)
    _log(node_config.rank, f"1a. Model '{model_name}' appears to be cached: {is_cached}")

    # Log memory before loading
    mem_before = psutil.virtual_memory()
    _log(node_config.rank, f"1b. Memory before loading: {mem_before.used / (1024**3):.2f} GB used / {mem_before.total / (1024**3):.2f} GB total")

    _log(node_config.rank, "1c. Loading full model with from_pretrained...")
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    _log(node_config.rank, "1d. Model object created.")

    # Log memory after loading
    mem_after = psutil.virtual_memory()
    _log(node_config.rank, f"1e. Memory after loading: {mem_after.used / (1024**3):.2f} GB used / {mem_after.total / (1024**3):.2f} GB total")
    _log(node_config.rank, f"   -> Memory consumed by model load: {(mem_after.used - mem_before.used) / (1024**3):.2f} GB")

    _log(node_config.rank, "2. Deep-copying model structure...")
    import copy
    model = copy.deepcopy(full_model)
    _log(node_config.rank, "2. Deep-copy complete.")

    _log(node_config.rank, "3. Pruning unused layers...")
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

    _log(node_config.rank, "3. Pruning complete.")

    _log(node_config.rank, "4. Deleting full model to free memory...")
    del full_model, all_layers, layers_to_keep
    _log(node_config.rank, "4. Deletion complete.")

    _log(node_config.rank, f"5. Moving sharded model to device '{device}'...")
    model = model.to(device)
    _log(node_config.rank, "5. Move to device complete.")

    return model

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