import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
from mamba_ssm.utils.generation import InferenceParams

from config import BackboneConfig


class ZonosBackbone(nn.Module):
    """
    Zonos 骨干网络（Backbone）类。

    该类实现了 Zonos 模型的骨干网络部分，包含多个层（blocks），每个层可能包含自注意力机制和前馈网络（MLP）。
    """
    def __init__(self, config: BackboneConfig):
        """
        初始化 Zonos 骨干网络。

        参数:
            config (BackboneConfig): 配置参数，包含模型的各种超参数。
        """
        super().__init__()
        # 保存配置参数
        self.config = config

        # 创建骨干网络的各个层（blocks）
        # 每个层根据是否在注意力层索引列表中，使用不同的中间层维度
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,  # 模型维度
                    d_intermediate=config.d_intermediate  # 中间层维度，
                    if (i not in config.attn_layer_idx)   # 如果当前层是注意力层，则使用 attn_mlp_d_intermediate；
                    else config.attn_mlp_d_intermediate,  # 否则，使用 d_intermediat
                    ssm_cfg=config.ssm_cfg,  # SSM 配置
                    layer_idx=i,  # 当前层的索引
                    attn_layer_idx=config.attn_layer_idx,  # 注意力层的索引列表
                    attn_cfg=config.attn_cfg,  # 注意力层的配置
                    norm_epsilon=config.norm_epsilon,  # 层归一化的 epsilon
                    residual_in_fp32=config.residual_in_fp32,  # 是否在 FP32 中保留残差
                    fused_add_norm=True,  # 是否融合加法和归一化操作
                    rms_norm=config.rms_norm,  # 是否使用 RMS 归一化
                )
                for i in range(config.n_layer)  # 根据层数创建相应数量的层
            ]
        )

        # 创建最终的层归一化层
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams | None = None):
        """
        前向传播方法。

        该方法对输入的隐藏状态进行逐层处理，并应用最终的层归一化。

        参数:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (batch, seq_len, d_model)。
            inference_params (InferenceParams, 可选): 推理参数，默认为 None。

        返回:
            torch.Tensor: 输出的隐藏状态，形状为 (batch, seq_len, d_model)。
        """
        # 初始化残差为 None
        residual = None

        # 逐层处理隐藏状态
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        # 应用最终的层归一化
        return layer_norm_fn(
            hidden_states,  # 输入的隐藏状态
            self.norm_f.weight,  # 层归一化的权重
            self.norm_f.bias,  # 层归一化的偏置
            residual,  # 残差
            eps=self.norm_f.eps,  # 层归一化的 epsilon
            residual_in_fp32=self.config.residual_in_fp32,  # 是否在 FP32 中保留残差
            is_rms_norm=self.config.rms_norm,  # 是否使用 RMS 归一化
        )
