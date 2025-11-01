import torch
import torch.nn as nn
import math


class Conv1DProjection(nn.Module):
    """使用1D卷积捕获时序模式的投影层"""

    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),  # 捕获局部模式
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(d_model),  # 自适应池化到目标维度
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, S)
        B, C, S = x.shape
        x = x.reshape(B * C, 1, S)  # (B*C, 1, S)
        x = self.conv(x)  # (B*C, 128, d_model)
        x = x.mean(dim=1)  # (B*C, d_model)
        x = x.reshape(B, C, -1)  # (B, C, d_model)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.use_ac = True  # ablation parameter, default: True
        self.channel_aggre = True  # ablation parameter, default: True

        # 模块一：稳定模式 - 学习每个变量的周期性模式
        # Θ_stable: 形状 (cycle_len, enc_in)
        if self.use_ac:
            self.temporalQuery = torch.nn.Parameter(
                torch.randn(self.cycle_len, self.enc_in) * 0.02,
                requires_grad=True
            )

        # 模块二：AC块 - 非对称跨变量注意力的投影层
        # W_Q, W_K, W_V: 将长度为seq_len的序列投影到d_model维
        if self.channel_aggre:
            self.W_Q = Conv1DProjection(self.seq_len, self.d_model, self.dropout)
            self.W_K = Conv1DProjection(self.seq_len, self.d_model, self.dropout)
            self.W_V = Conv1DProjection(self.seq_len, self.d_model, self.dropout)

        # 输入投影层
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        # 前馈网络
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, cycle_index):
        """
        Args:
            x: 输入序列，形状 (B, S, C) 其中 B=batch_size, S=seq_len, C=enc_in
            cycle_index: 当前时间在周期中的位置，形状 (B,)

        Returns:
            output: 预测结果，形状 (B, pred_len, C)
        """

        # Instance Normalization (RevIN)
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, C)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # (B, 1, C)
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 转置：(B, S, C) -> (B, C, S)
        # 这样每个变量的完整时间序列在最后一维
        x_input = x.permute(0, 2, 1)  # (B, C, S)

        # ========== AC块：非对称跨变量注意力 ==========
        if self.use_ac and self.channel_aggre:
            # ===== 步骤1：生成查询Q（稳定的提问）=====
            # 根据当前时间t从稳定模式Θ_stable中提取周期性片段P_t
            # gather_index: (B, S) - 每个样本在每个时间步对应的周期位置
            gather_index = (
                                   cycle_index.view(-1, 1) +
                                   torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)
                           ) % self.cycle_len

            # P_t: (B, S, C) - 从temporalQuery中提取对应的周期性片段
            P_t = self.temporalQuery[gather_index]

            # 转置为 (B, C, S)
            P_t = P_t.permute(0, 2, 1)

            # Q = P_t @ W_Q: (B, C, S) -> (B, C, d_model)
            Q = self.W_Q(P_t)

            # ===== 步骤2：生成键K和值V（动态的真实数据）=====
            # K = X_in @ W_K: (B, C, S) -> (B, C, d_model)
            K = self.W_K(x_input)

            # V = X_in @ W_V: (B, C, S) -> (B, C, d_model)
            V = self.W_V(x_input)

            # ===== 步骤3：执行跨变量注意力计算 =====
            # 计算注意力得分矩阵: Q @ K^T
            # (B, C, d_model) @ (B, d_model, C) -> (B, C, C)
            # 这是文档中的核心：(C, C)的注意力矩阵直接捕获变量间的空间依赖
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

            # Softmax归一化，得到注意力权重
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, C, C)

            # 应用注意力权重到值V
            # (B, C, C) @ (B, C, d_model) -> (B, C, d_model)
            channel_information = torch.matmul(attn_weights, V)

        elif not self.use_ac and self.channel_aggre:
            # 消融实验：不使用稳定模式，但使用跨变量注意力
            # Q, K, V都来自原始输入x_input（对称注意力）
            Q = self.W_Q(x_input)  # (B, C, d_model)
            K = self.W_K(x_input)  # (B, C, d_model)
            V = self.W_V(x_input)  # (B, C, d_model)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            channel_information = torch.matmul(attn_weights, V)

        elif self.use_ac and not self.channel_aggre:
            # 消融实验：使用稳定模式但不做跨变量注意力
            # 直接使用稳定模式作为额外信息
            gather_index = (
                                   cycle_index.view(-1, 1) +
                                   torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)
                           ) % self.cycle_len
            P_t = self.temporalQuery[gather_index]  # (B, S, C)
            P_t = P_t.permute(0, 2, 1)  # (B, C, S)
            channel_information = self.input_proj(P_t)  # (B, C, d_model)

        else:
            # 消融实验：不使用任何跨变量信息
            channel_information = 0

        # 输入投影
        x_proj = self.input_proj(x_input)  # (B, C, d_model)

        # 残差连接：加上跨变量信息
        input_combined = x_proj + channel_information

        # 前馈网络
        hidden = self.model(input_combined)  # (B, C, d_model)

        # 残差连接 + 输出投影
        output = self.output_proj(hidden + input_combined)  # (B, C, pred_len)

        # 转置回原始格式：(B, C, pred_len) -> (B, pred_len, C)
        output = output.permute(0, 2, 1)

        # Instance Denormalization (RevIN)
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output