import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np


class VTILayer(nn.Module):
    # 新增 target_means 参数，默认为 None 以兼容旧代码
    def __init__(self, vti_direction, lam, target_means=None):
        super(VTILayer, self).__init__()
        self.vti_direction = vti_direction
        self.lam = lam
        self.target_means = target_means 

    def forward(self, x):
        if self.vti_direction is not None:
            norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)            
            y = 0
            for i in range(len(self.vti_direction)):
                # 获取单位方向向量 u
                u = F.normalize(self.vti_direction[i], dim=-1)
                
                # --- 动态强度核心逻辑 ---
                if self.target_means is not None:
                    # 计算当前投影: proj = x^T * u
                    current_proj = torch.sum(x.float() * u, dim=-1, keepdim=True)
                    target = self.target_means[i]
                    diff = target - current_proj

                    # 计算相对误差：(差值 / 目标值的绝对值)
                    # 加上 1e-6 防止 target 为 0 (虽然你的数据里没 0，但保险起见)
                    relative_diff = diff / (torch.abs(target) + 1e-6)

                    # 乘一个系数让它在 Sigmoid 敏感区，比如 2.0 或 5.0
                    lambda_sim = torch.sigmoid(diff)
                else:
                    lambda_sim = 1.0 

                # 根据是否 Decoding 阶段调整形状
                if x.size(1) < 2:
                    y += self.lam[i] * lambda_sim * u.repeat(1, x.shape[1], 1)
                else:
                    y += self.lam[i] * lambda_sim * u
            
            y = y / len(self.vti_direction)
            # 注入干预并归一化
            x = F.normalize(F.normalize(x.float(), dim=-1) + 0.1 * y, dim=-1) * norm
                
            return x.half()
        else:
            return x


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_vti_layers(model: PreTrainedModel, vti_directions: Tensor, alpha: list, target_means: Tensor = None, layers_range: tuple = None):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]

    assert len(vti_directions) == len(layers)
    for i, layer in enumerate(layers):

        if layers_range is not None:
            if i < layers_range[0] or i > layers_range[1]:
                continue

        original_mlp = find_module(layer, mlp_keywords)
        
        current_mean = None
        if target_means is not None:
            # target_means[i] 取出来是标量 (0-dim) 我们用 .view(1) 把它变回 [1] 的形状 (1-dim)
            current_mean = target_means[i].view(1) 
        
        layer.mlp = nn.Sequential(
            original_mlp, 
            VTILayer(vti_directions[i], alpha, target_means=current_mean)
        )

def remove_vti_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        vti_mlp = find_module(layer, mlp_keywords)
        if isinstance(vti_mlp, nn.Sequential):
            layer.mlp = vti_mlp[0]