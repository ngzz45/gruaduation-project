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

                    
                    # # 计算相对误差：(差值 / 目标值的绝对值)
                    # relative_diff = diff / (torch.abs(target) + 1e-6)

                    lambda_sim = torch.clamp(diff, min=0, max=1.0)
                    # print(f"Layer lambda_sim: {diff.mean().item()}")
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
            # x = x.float() + y

            return x.half()
        else:
            return x

class VTIBlockWrapper(nn.Module):
    def __init__(self, original_layer, vti_layer):
        super().__init__()
        self.original_layer = original_layer
        self.vti_layer = vti_layer # 这就是你之前写的 VTILayer 实例

    def forward(self, *args, **kwargs):
        # 1. 先让原始 Block 跑完，拿到结果
        # outputs 通常是 tuple: (hidden_states, self_attentions, ...)
        outputs = self.original_layer(*args, **kwargs)
        
        # 2. 取出残差流 (Hidden States)
        # 它是 Block 的最终输出，也就是下一层的输入
        hidden_states = outputs[0]
        
        # 3. 在残差流上应用 VTI 干预
        # 此时传入 vti_layer 的 x 就是真正的 residual stream vector
        hidden_states = self.vti_layer(hidden_states)
        
        # 4. 把修改后的 hidden_states 塞回 tuple
        # 保持输出格式与原始模型一致，防止报错
        return (hidden_states,) + outputs[1:]

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
            if i+1 < layers_range[0] or i+1 > layers_range[1]:
                continue
 
        
        current_mean = None
        if target_means is not None:
            # target_means[i] 取出来是标量 (0-dim) 我们用 .view(1) 把它变回 [1] 的形状 (1-dim)
            current_mean = target_means[i].view(1) 
            
        vti_layer_instance = VTILayer(
            vti_directions[i], 
            alpha, 
            target_means=current_mean
        )
        
        # original_mlp = find_module(layer, mlp_keywords)        
        # layer.mlp = nn.Sequential(
        #     original_mlp, 
        #     vti_layer_instance
        # )

        layers[i] = VTIBlockWrapper(layer, vti_layer_instance)

def remove_vti_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        vti_mlp = find_module(layer, mlp_keywords)
        if isinstance(vti_mlp, nn.Sequential):
            layer.mlp = vti_mlp[0]