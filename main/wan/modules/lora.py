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
        self.enabled = True  # 默认启用LoRA
        
        # Initialize low-rank matrices with proper initialization
        self.lora_A = nn.Parameter(torch.zeros(layer.weight.size(0), rank),requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, layer.weight.size(1)),requires_grad=True)

        # 使用 Kaiming 初始化 A 矩阵
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 矩阵初始化为零，确保训练开始时 LoRA 无贡献
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始层的输出
        original_output = self.layer(x)
        
        # print(f"x shape: {x.shape}, original_output shape: {original_output.shape} layer weight shape: {self.layer.weight.shape}")
        
        # 只有在enabled为True时才应用LoRA
        if self.enabled:
            # 保存原始形状信息
            original_shape = x.shape
            
            if len(original_shape) == 3:
                # 三维输入: [batch_size, seq_len, hidden_dim]
                batch_size, seq_len, hidden_dim = original_shape
                
                # 重塑为二维张量来执行LoRA计算
                x_2d = x.reshape(-1, hidden_dim)  # [batch_size*seq_len, hidden_dim]
                
                # 执行低秩适配
                lora_output = (x_2d @ self.lora_B.T) @ self.lora_A.T
                
                # 重塑回原始形状
                lora_output = lora_output.reshape(batch_size, seq_len, -1)
                
            else:
                # 对于二维输入保持原有逻辑
                lora_output = x @ self.lora_B.T @ self.lora_A.T
            
            # 缩放输出并添加到原始输出
            return original_output + self.scale * lora_output
        else:
            # 不使用LoRA，直接返回原始输出
            return original_output

    def set_lora_enabled(self, enabled=True,scale=None):
        """设置是否启用LoRA"""
        self.enabled = enabled
        if scale is not None:
            self.scale = scale


def apply_lora_to_wanx(model, rank=4, scale=1.0):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ((name.endswith('.q') or name.endswith('.k') or name.endswith('.v') or name.endswith('.o'))):
            modules_to_replace.append((name, module))
    
    count = 0
    for name, module in modules_to_replace:
        lora_layer = LoRALayer(module, rank, scale)
        # 正确处理嵌套模块
        name_parts = name.split('.')
        parent_module = model
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], lora_layer)
        count += 1
    print(f'Applied LoRA to {count} layers')
    
    return model


# 新增：在整个模型范围内设置LoRA状态
def set_lora_state(model, enabled=True, requires_grad=None, scale=None):
    """
    在整个模型中设置所有LoRA层的状态
    
    Args:
        model: 包含LoRA层的模型
        enabled: 是否启用LoRA,True为启用,False为禁用
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.set_lora_enabled(enabled, scale)
            if requires_grad is not None:
                module.lora_A.requires_grad = requires_grad
                module.lora_B.requires_grad = requires_grad
            module.layer.requires_grad_(False)
            count += 1
    
    # state = "启用" if enabled else "禁用"
    # print(f"已{state} {count} 个LoRA层")
    return model


if __name__ == '__main__':
    # Test LoRALayer
    from main.wan.modules.model import *
    model = WanModel()
    model = apply_lora_to_wanx(model)
    
    # 测试LoRA开关功能
    print("使用LoRA进行推理...")
    # ... 执行推理代码 ...
    
    print("禁用LoRA...")
    set_lora_state(model, enabled=False)
    # ... 执行不带LoRA的推理代码 ...
    
    print("重新启用LoRA...")
    set_lora_state(model, enabled=True)
    # ... 执行带LoRA的推理代码 ...