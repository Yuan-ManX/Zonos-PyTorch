import torch
import torch.nn.functional as F


# 定义函数 apply_delay_pattern，用于在编码序列中应用延迟模式
def apply_delay_pattern(codes: torch.Tensor, mask_token: int):
    """
    在编码序列中应用延迟模式。

    该函数通过在编码序列的右侧填充掩码标记（mask_token），然后对每个时间步的编码进行循环移位，实现延迟效果。

    参数:
        codes (torch.Tensor): 输入的编码序列，形状为 (batch, seq_len)。
            - `batch`: 批次大小。
            - `seq_len`: 序列长度。
        mask_token (int): 用于填充的掩码标记。

    返回:
        torch.Tensor: 应用延迟模式后的编码序列，形状为 (batch, seq_len + padding)。
            - `padding`: 填充的长度，等于原始序列长度。
    """
    # 在编码序列的右侧填充掩码标记，填充长度为原始序列长度
    # 新的序列形状为 (batch, seq_len + seq_len)
    codes = F.pad(codes, (0, codes.shape[1]), value=mask_token)

    # 对填充后的序列进行循环移位操作
    # 对每个时间步 k，将序列向右循环移位 k + 1 位
    # 结果是一个新的张量，形状为 (batch, seq_len, seq_len + seq_len)
    # 返回应用延迟模式后的编码序列
    return torch.stack([codes[:, k].roll(k + 1) for k in range(codes.shape[1])], dim=1)


# 定义函数 revert_delay_pattern，用于恢复应用延迟模式前的编码序列
def revert_delay_pattern(codes: torch.Tensor):
    """
    恢复应用延迟模式前的编码序列。

    该函数通过对延迟后的编码序列进行切片操作，恢复原始的编码序列。

    参数:
        codes (torch.Tensor): 应用延迟模式后的编码序列，形状为 (batch, n_q, seq_len)。
            - `batch`: 批次大小。
            - `n_q`: 延迟模式中引入的查询数量。
            - `seq_len`: 序列长度。

    返回:
        torch.Tensor: 恢复后的编码序列，形状为 (batch, seq_len - n_q)。
    """
    # 获取张量的维度
    _, n_q, seq_len = codes.shape

    # 对每个批次和查询，对编码序列进行切片
    # 从第 (k + 1) 个时间步开始，切片到 (seq_len - n_q + k + 1) 个时间步
    # 结果是一个新的张量，形状为 (batch, n_q, seq_len - n_q)
    # 返回恢复后的编码序列
    return torch.stack([codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], dim=1)
