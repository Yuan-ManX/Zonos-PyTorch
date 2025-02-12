import torch


def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """
    对输入张量进行多项式抽样，支持任意维度的输入，最后一维为候选数量。

    参数:
        input (torch.Tensor): 包含概率分布的输入张量。
        num_samples (int): 要抽取的样本数量。
        replacement (bool, 可选): 是否进行有放回抽样，默认为 False（无放回）。
        generator (torch.Generator, 可选): 用于抽样的伪随机数生成器，默认为 None。

    返回:
        torch.Tensor: 最后一维包含从输入张量的多项式概率分布中抽取的 num_samples 个索引的张量。
    """

    if num_samples == 1:
        # 如果只需要抽取一个样本，则使用指数分布生成随机数，并取 argmax 作为结果
        # 生成与 input 形状相同的指数分布随机数
        q = torch.empty_like(input).exponential_(1, generator=generator)
        # 计算 input / q 的 argmax，并转换为 int64 类型
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)
    
    # 如果需要抽取多个样本，则将输入张量重塑为二维张量，最后一维为候选数量
    input_ = input.reshape(-1, input.shape[-1])
    # 使用 torch.multinomial 进行多项式抽样
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    # 将输出张量重塑回原始输入的形状，最后一维为 num_samples
    output = output_.reshape(*list(input.shape[:-1]), -1)
    # 返回抽样结果
    return output


def apply_top_k(
    probs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    从输入概率张量的最后一维中抽取 top-k 值，并进行概率归一化。

    参数:
        probs (torch.Tensor): 输入概率张量，最后一维为候选 token。
        k (int): top-k 中的 k 值。

    返回:
        torch.Tensor: 归一化后的概率张量，仅保留 top-k 概率，其余概率设为0。
    """
    # 获取 top-k 概率和对应的索引
    v, _ = torch.topk(probs, min(k, probs.size(-1)))
    # 获取 top-k 中的最小概率值
    pivot = v.select(-1, -1).unsqueeze(-1)
    # 将小于最小 top-k 概率的概率设为0
    probs = torch.where(probs < pivot, 0.0, probs)
    # 对剩余概率进行归一化
    probs.div_(probs.sum(dim=-1, keepdim=True))
    # 返回处理后的概率张量
    return probs


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    从输入概率张量的最后一维中抽取 top-p 概率，并进行概率归一化。

    参数:
        probs (torch.Tensor): 输入概率张量，最后一维为候选 token。
        p (float): top-p 中的 p 值。

    返回:
        torch.Tensor: 归一化后的概率张量，仅保留 top-p 概率，其余概率设为0。
    """
    # 对概率进行排序，降序排列
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算累积和
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 创建掩码，标记累积和超过 p 的位置
    mask = probs_sum - probs_sort > p
    # 将超过 p 的概率设为0
    probs_sort *= (~mask).float()
    # 将排序后的概率重新赋值回原始张量
    probs = probs.scatter(-1, probs_idx, probs_sort)
    # 对剩余概率进行归一化
    probs.div_(probs.sum(dim=-1, keepdim=True))
    # 返回处理后的概率张量
    return probs


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    使用最小概率（min-p）进行下一个 token 的采样。

    参数:
        probs (torch.Tensor): 输入的概率张量，最后一维为候选 token。
        min_p (float): 最小 token 概率，相对于最可能 token 的概率进行缩放。
                      必须介于 0 和 1 之间。典型值在 0.01 到 0.2 之间。

    返回:
        torch.Tensor: 处理后的概率张量，低于 min_p 的概率被设为 0，然后进行归一化。
    """
    # 获取每个样本的最大概率，并保持维度以便后续广播
    top_probs, _ = probs.max(dim=-1, keepdim=True)

    # 计算需要移除的 token 掩码：概率小于 (min_p * 最大概率)
    tokens_to_remove = probs < (min_p * top_probs)

    # 将需要移除的 token 的概率设为 0
    probs = probs.masked_fill(tokens_to_remove, 0.0)

    # 对剩余的概率进行归一化
    probs.div_(probs.sum(dim=-1, keepdim=True))

    # 返回处理后的概率张量
    return probs


def modify_logit_for_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    repetition_penalty: float,
    repetition_penalty_window: int,
):
    """
    在最后 `repetition_penalty_window` 个 token 上应用重复惩罚。

    参数:
        logits (torch.Tensor): 输入的 logits 张量，形状为 (batch_size, n_codebooks, vocab_size)。
        generated_tokens (torch.Tensor): 生成的 token，形状为 (batch_size, n_codebooks, seq_len)。
        repetition_penalty (float): 重复惩罚因子。值大于1会降低重复 token 的概率，值小于1会提高重复 token 的概率。
        repetition_penalty_window (int): 重复惩罚窗口大小，即考虑最后多少个生成的 token 进行惩罚。

    返回:
        torch.Tensor: 修改后的 logits 张量。
    """
    # 获取最后 `repetition_penalty_window` 个生成的 token
    generated_tokens = generated_tokens[..., -repetition_penalty_window:]

    # 将生成的 token 限制在词汇表范围内，并转换为 int64 类型
    generated_tokens = generated_tokens.clamp_max(logits.shape[-1] - 1).to(torch.int64)

    # 创建一个与 logits 形状相同的重复惩罚因子张量，初始值为 repetition_penalty
    rp = torch.full_like(logits, repetition_penalty)

    # 对生成的 token 应用重复惩罚因子，使用乘积方式
    factors = torch.ones_like(logits).scatter_reduce(2, generated_tokens, rp, reduce="prod")

    # 根据 logits 的正负情况，应用不同的重复惩罚
    # 如果 logits <= 0，则乘以 factors；否则，除以 factors
    return torch.where(logits <= 0, logits * factors, logits / factors)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    generated_tokens: torch.Tensor | None = None,
    repetition_penalty: float = 3.0,
    repetition_penalty_window: float = 2,
) -> torch.Tensor:
    """
    从 logits 中使用温度、top-p、top-k 或 min-p 采样方法采样下一个 token。

    参数:
        logits (torch.Tensor): 输入的 logits 张量，最后一维为候选 token。
        temperature (float, 可选): 采样温度，默认为1.0。较低的温度会使采样更确定。
        top_p (float, 可选): top-p 值，默认为0.0。
        top_k (int, 可选): top-k 值，默认为0。
        min_p (float, 可选): 最小概率值，默认为0.0。必须介于0和1之间，典型值在0.01到0.2之间。
        generated_tokens (torch.Tensor, 可选): 已生成的 token，默认为 None。
        repetition_penalty (float, 可选): 重复惩罚因子，默认为3.0。
        repetition_penalty_window (float, 可选): 重复惩罚窗口大小，默认为2。

    返回:
        torch.Tensor: 采样的 token，形状为 (batch_size, num_codebooks, 1)。
    """
    # 如果启用了重复惩罚，并且有已生成的 token，则应用重复惩罚
    if repetition_penalty != 1.0 and generated_tokens is not None:
        logits = modify_logit_for_repetition_penalty(logits, generated_tokens, repetition_penalty, repetition_penalty_window)

    if temperature > 0:
        # 应用温度到 logits 上，然后计算概率
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p > 0:
            # 应用 top-p 采样
            probs = apply_top_p(probs, top_p)
        if top_k > 0:
            # 应用 top-k 采样
            probs = apply_top_k(probs, top_k)
        if min_p > 0:
            # 应用 min-p 采样
            probs = apply_min_p(probs, min_p)

        # 从概率分布中采样下一个 token
        next_token = multinomial(probs, num_samples=1)
    else:
        # 如果温度为0，则直接选择 logits 最大的 token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    # 返回采样的 token，形状为 (batch_size, num_codebooks, 1)
    return next_token
