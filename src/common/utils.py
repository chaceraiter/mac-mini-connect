"""
Utility functions for the project.
"""
from torch import nn
import datetime

def _log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
    print(f"    [{timestamp}] {message}", flush=True)

def get_nested_attr(obj, path):
    """
    Get a nested attribute from an object using a dot-separated path.
    Example: get_nested_attr(model, 'transformer.h.0.attn')
    """
    _log(f"[get_nested_attr] Trying to get path: {path}")
    parts = path.split('.')
    for part in parts:
        if isinstance(obj, nn.ModuleList) and part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

def set_nested_attr(obj, path, value):
    """
    Set a nested attribute on an object using a dot-separated path.
    Example: set_nested_attr(model, 'transformer.h.0.attn', new_attn)
    """
    _log(f"[set_nested_attr] Trying to set path: {path}")
    parts = path.split('.')
    for part in parts[:-1]:
        if isinstance(obj, nn.ModuleList) and part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    
    last_part = parts[-1]
    if isinstance(obj, nn.ModuleList) and last_part.isdigit():
        obj[int(last_part)] = value
    else:
        setattr(obj, last_part, value) 