import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim=5120, num_classes=1, hidden_dim=1280, num_heads=8, 
                 num_conv_layers=3, dropout=0.1, kernel_size=5, num_additional_blocks=2):
        super(Classifier, self).__init__()
        
        # 卷积层和池化层，用于减少序列长度
        self.conv_layers = nn.ModuleList()
        
        # 第一个卷积层 - 将输入通道数转换为hidden_dim
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            )
        )
        
        # 添加更多卷积层
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout)
                )
            )
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        
        # 添加额外的卷积+注意力块
        self.additional_blocks = nn.ModuleList()
        for i in range(num_additional_blocks):
            # 每个块包含一个卷积层用于减少序列长度
            conv_block = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            )
            
            # 每个块包含一个注意力层
            attn_layer = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
            
            # 每个块包含两个层归一化
            norm1 = nn.LayerNorm(hidden_dim)
            norm2 = nn.LayerNorm(hidden_dim)
            
            # 每个块包含一个前馈网络
            ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            
            self.additional_blocks.append(nn.ModuleDict({
                'conv': conv_block,
                'attention': attn_layer,
                'norm1': norm1,
                'norm2': norm2,
                'ffn': ffn
            }))
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 输出分类层
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        输入: x 形状 [batch_size, seq_len, dim]
        输出: 形状 [batch_size, num_classes]
        """
        # 调换维度顺序用于卷积 [batch_size, dim, seq_len]
        x = x.transpose(1, 2)
        
        # 应用卷积层减少序列长度
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 调换回注意力机制所需的维度顺序 [batch_size, new_seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # 自注意力机制 (带残差连接)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # 添加额外的卷积+注意力块处理
        for block in self.additional_blocks:
            # 调换维度用于卷积 [batch_size, hidden_dim, seq_len]
            x_conv = x.transpose(1, 2)
            # 应用卷积减少序列长度
            x_conv = block['conv'](x_conv)
            # 调换回 [batch_size, seq_len, hidden_dim]
            x_conv = x_conv.transpose(1, 2)
            
            # 应用注意力机制
            attn_output, _ = block['attention'](x_conv, x_conv, x_conv)
            x = block['norm1'](x_conv + attn_output)
            
            # 应用前馈网络
            ffn_output = block['ffn'](x)
            x = block['norm2'](x + ffn_output)

        
        # 前馈网络 (带残差连接)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # 序列聚合 - 使用平均池化
        x = torch.mean(x, dim=1)
        
        # 分类层
        output = self.classifier(x)
        
        return output


# 测试用例
if __name__ == "__main__":
    device = torch.cuda.current_device()
    model = Classifier().to(device).to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dummy_input = torch.randn(1, 32760, 5120, dtype=torch.bfloat16, device=device)
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计
        output = model(dummy_input)
        target = torch.randn(1, 1, dtype=torch.bfloat16, device=device)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        loss.backward()
    print(f"最大显存使用量: {torch.cuda.max_memory_reserved()/ 1024**3:.2f} GB")
    print(f"Loss: {loss.item()}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")