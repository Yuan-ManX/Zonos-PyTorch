from dataclasses import dataclass, field
from typing import Literal


# 定义 BackboneConfig 数据类，用于配置骨干网络（Backbone）的参数
@dataclass
class BackboneConfig:
    """
    骨干网络配置类。

    该类包含骨干网络的各种超参数，用于定义骨干网络的结构和行为。
    """
    d_model: int = 1024   # 模型维度，默认为1024
    d_intermediate: int = 0   # 中间层的维度，默认为0
    attn_mlp_d_intermediate: int = 0   # 注意力层中 MLP 的中间层维度，默认为0
    n_layer: int = 16   # 网络层数，默认为16层
    ssm_cfg: dict = field(default_factory=dict)   # SSM（状态空间模型）配置，默认为空字典
    attn_layer_idx: list = field(default_factory=list)   # 注意力层的索引列表，默认为空列表
    attn_cfg: dict = field(default_factory=dict)   # 注意力层的配置，默认为空字典
    rms_norm: bool = False   # 是否使用 RMS 归一化，默认为 False
    residual_in_fp32: bool = False   # 是否在 FP32 中保留残差，默认为 False
    norm_epsilon: float = 1e-5   # 层归一化的 epsilon，默认为1e-5


# 定义 PrefixConditionerConfig 数据类，用于配置前缀条件器（Prefix Conditioner）的参数
@dataclass
class PrefixConditionerConfig:
    """
    前缀条件器配置类。

    该类包含前缀条件器的各种配置参数，用于定义前缀条件器的行为。
    """
    conditioners: list[dict]  # 条件器的列表，每个条件器由一个字典定义
    projection: Literal["none", "linear", "mlp"]  # 投影方式，可以是 "none"（无投影）、"linear"（线性投影）或 "mlp"（多层感知机投影）


# 定义 ZonosConfig 数据类，用于配置 Zonos 模型的参数
@dataclass
class ZonosConfig:
    """
    Zonos 模型配置类。

    该类包含 Zonos 模型的各种配置参数，涵盖骨干网络、前缀条件器以及其他相关参数。
    """
    backbone: BackboneConfig  # 骨干网络配置
    prefix_conditioner: PrefixConditionerConfig  # 前缀条件器配置
    eos_token_id: int = 1024  # 序列结束标记的 ID，默认为1024
    masked_token_id: int = 1025  # 掩码标记的 ID，默认为1025

    @classmethod
    def from_dict(cls, d: dict) -> "ZonosConfig":
        """
        从字典创建 ZonosConfig 实例。

        该类方法从给定的字典中提取配置参数，并创建 ZonosConfig 实例。

        参数:
            d (dict): 包含配置参数的字典。

        返回:
            ZonosConfig: 创建的 ZonosConfig 实例。
        """
        # 复制输入字典以避免修改原始数据
        d = d.copy()

        # 从字典中提取骨干网络配置，并创建 BackboneConfig 实例
        backbone_config = BackboneConfig(**d.pop("backbone"))
        # 从字典中提取前缀条件器配置，并创建 PrefixConditionerConfig 实例
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))

        # 使用提取的配置参数创建 ZonosConfig 实例
        config = cls(backbone_config, prefix_conditioner_config, **d)

        # 返回创建的 ZonosConfig 实例
        return config
