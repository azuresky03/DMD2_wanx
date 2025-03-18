# filepath: /wan-lora/modules/lora.py

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, layer, rank=4, scale=1.0):
        super(LoRALayer, self).__init__()
        self.layer = layer
        self.rank = rank
        self.scale = scale
        
        # Initialize low-rank matrices with proper initialization
        self.lora_A = nn.Parameter(torch.zeros(layer.weight.size(0), rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, layer.weight.size(1)))

        # 使用 Kaiming 初始化 A 矩阵
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 矩阵初始化为零，确保训练开始时 LoRA 无贡献
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Compute the low-rank adaptation
        lora_output = (self.lora_A @ self.lora_B) @ x
        # Scale the output
        return self.layer(x) + self.scale * lora_output

def apply_lora_to_wanx(model, rank=4, scale=1.0):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ((name.endswith('.q') or name.endswith('.k') or name.endswith('.v') or name.endswith('.o'))):
            modules_to_replace.append((name, module))
    
    count = 0
    for name, module in modules_to_replace:
        lora_layer = LoRALayer(module, rank, scale)
        setattr(model, name, lora_layer)
        count += 1
    print(f'Applied LoRA to {count} layers')
    
    return model


if __name__ == '__main__':
    # Test LoRALayer
    from main.wan.modules.model import *
    model = WanModel()
    model = apply_lora_to_wanx(model)