import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim=5120):
        super().__init__()
        
        self.main = nn.Sequential(
            # 输入形状: [batch, 5120, 32760]
            nn.Conv1d(input_dim, 5120, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 5120),
            nn.SiLU(),
            
            nn.Conv1d(5120, 5120, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 5120),
            nn.SiLU(),
            
            nn.Conv1d(5120, 5120, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 5120),
            nn.SiLU(),
            
            # 最终下采样层
            nn.Conv1d(5120, 5120//4, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(32, 5120//4),
            nn.SiLU(),
            
            # 全局信息聚合
            nn.AdaptiveAvgPool1d(1),  # 输出形状: [batch, 1280, 1]
            nn.Flatten(),             # 输出形状: [batch, 1280]
            
            # 分类头
            nn.Linear(5120//4, 1)     # 二分类输出一个logit
        )

    def forward(self, x):
        # 输入形状转换: [B, Seq_len, Feat] => [B, Feat, Seq_len]
        x = x.transpose(1, 2)
        return self.main(x)

# 测试用例
if __name__ == "__main__":
    model = Classifier()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        dummy_input = torch.randn(1, 32760, 5120)  # 输入形状
        output = model(dummy_input)                # 输出形状: [1, 1]
        target = torch.randn(1,1)                    # 随机目标
        criterion = nn.BCEWithLogitsLoss()          # 二分类交叉熵
        loss = criterion(output, target)            # 计算损失
        loss.backward()                             # 反向传播
    print(f"Loss: {loss.item()}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")