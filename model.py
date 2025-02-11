import json
import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from mamba_ssm.utils.generation import InferenceParams
from tqdm import trange

from autoencoder import DACAutoencoder
from backbone import ZonosBackbone
from codebook_pattern import apply_delay_pattern, revert_delay_pattern
from conditioning import PrefixConditioner
from config import ZonosConfig
from sampling import sample_from_logits
from speaker_cloning import SpeakerEmbeddingLDA


class Zonos(nn.Module):
    """
    Zonos 模型类。

    该类实现了 Zonos 模型，集成了自动编码器、骨干网络、前缀条件器和说话人嵌入模型等功能。
    """
    def __init__(self, config: ZonosConfig):
        """
        初始化 Zonos 模型。

        参数:
            config (ZonosConfig): Zonos 模型的配置参数。
        """
        super().__init__()

        # 保存配置参数
        self.config = config
        # 获取模型的维度
        dim = config.backbone.d_model
        # 获取序列结束标记的 ID
        self.eos_token_id = config.eos_token_id
        # 获取掩码标记的 ID
        self.masked_token_id = config.masked_token_id

        # 初始化自动编码器
        self.autoencoder = DACAutoencoder()
        # 初始化骨干网络
        self.backbone = ZonosBackbone(config.backbone)
        # 初始化前缀条件器
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        # 初始化说话人克隆模型，默认为 None
        self.spk_clone_model = None

        # TODO: 填充到至少 8 的倍数
        # 初始化嵌入层列表，每个自动编码器的码本对应一个嵌入层
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        # 初始化线性层列表，每个自动编码器的码本对应一个线性层，用于生成最终输出
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        # 初始化缓存变量，用于条件生成（CG）
        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str | None = None, device: str = "cuda") -> "Zonos":
        """
        从预训练模型创建 Zonos 实例。

        该类方法从指定的 Hugging Face 仓库下载配置文件和模型权重，并加载到 Zonos 模型中。

        参数:
            repo_id (str): Hugging Face 仓库的 ID。
            revision (str, 可选): 模型的修订版本，默认为 None。
            device (str, 可选): 设备类型，默认为 "cuda"（GPU）。

        返回:
            Zonos: 加载了预训练权重的 Zonos 模型实例。
        """
        # 从仓库下载配置文件
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        # 从仓库下载模型权重文件
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        # 使用本地文件创建 Zonos 实例
        return cls.from_local(config_path, model_path, device)

    @classmethod
    def from_local(cls, config_path: str, model_path: str, device: str = "cuda") -> "Zonos":
        """
        从本地文件创建 Zonos 实例。

        该类方法从本地配置文件和模型权重文件加载 Zonos 模型。

        参数:
            config_path (str): 配置文件的路径。
            model_path (str): 模型权重文件的路径。
            device (str, 可选): 设备类型，默认为 "cuda"（GPU）。

        返回:
            Zonos: 加载了本地权重的 Zonos 模型实例。
        """
        # 从本地配置文件加载配置参数
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        # 创建 Zonos 模型实例，并移动到指定设备，使用 bfloat16 精度
        model = cls(config).to(device, torch.bfloat16)
        # 将自动编码器的 DAC 模型移动到指定设备
        model.autoencoder.dac.to(device)

        # 加载模型状态字典
        sd = model.state_dict()
        # 从本地模型权重文件加载权重
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        # 将加载的权重应用到模型中
        model.load_state_dict(sd)

        # 返回加载后的模型
        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        """
        从音频片段生成说话人嵌入。

        参数:
            wav (torch.Tensor): 输入音频张量。
            sr (int): 音频的采样率。

        返回:
            torch.Tensor: 生成的说话人嵌入。
        """
        if self.spk_clone_model is None:
            # 如果说话人克隆模型尚未初始化，则初始化 SpeakerEmbeddingLDA 模型
            self.spk_clone_model = SpeakerEmbeddingLDA()
        # 生成说话人嵌入
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        # 返回说话人嵌入，并转换为 bfloat16 精度
        return spk_embedding.unsqueeze(0).bfloat16()

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        对编码后的音频代码进行嵌入。

        参数:
            codes (torch.Tensor): 输入的编码音频代码，形状为 (batch, num_codebooks)。

        返回:
            torch.Tensor: 嵌入后的张量，形状为 (batch, dim)。
        """
        # 对每个码本应用嵌入层，并求和得到最终的嵌入
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        对隐藏状态应用线性层头部。

        参数:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (batch, dim)。

        返回:
            torch.Tensor: 应用头部后的输出，形状为 (batch, num_codebooks, 1025)。
        """
        # 对每个头部应用线性层，并将结果堆叠在新的维度上
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        """
        将 `hidden_states` 传递给 `backbone` 和 `multi_head`，如果 `cfg_scale != 1.0`，则应用分类器自由引导（CFG）。

        参数:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (batch, seq_len, dim)。
            inference_params (InferenceParams): 推理参数。
            cfg_scale (float): CFG 的缩放因子。

        返回:
            torch.Tensor: 计算得到的 logits，形状为 (batch, vocab_size)。
        """
        # 将隐藏状态传递给骨干网络，获取最后一个时间步的隐藏状态，并增加一个维度，形状为 (batch, 1, dim)
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        # 对最后一个时间步的隐藏状态应用多头线性层，生成 logits，并移除多余的维度，形状为 (batch, vocab_size)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()

        if cfg_scale != 1.0:
            # 如果 CFG 缩放因子不为1，则将 logits 拆分为有条件和无条件两部分
            cond_logits, uncond_logits = logits.chunk(2)
            # 应用 CFG 公式：logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        
        # 返回计算得到的 logits
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        """
        单步解码。准备隐藏状态，可能复制它们以用于 CFG，然后委托给 `_compute_logits` 进行计算。

        在此方法中，我们使用一个简单的 CUDA 图捕获机制，如果需要，执行3次预热步骤，然后捕获或重放图。
        只有在批大小发生变化时，我们才会重新捕获图。

        参数:
            input_ids (torch.Tensor): 输入的 token ID，形状为 (batch, seq_len)。
            inference_params (InferenceParams): 推理参数。
            cfg_scale (float): CFG 的缩放因子。

        返回:
            torch.Tensor: 解码得到的 logits，形状为 (batch, vocab_size)。
        """
        # TODO: 支持 cfg_scale == 1 的情况
        if cfg_scale == 1.0:
            # 如果 CFG 缩放因子为1，则直接嵌入输入 IDs 并计算 logits
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        # 获取批次大小
        bsz = input_ids.size(0)

        # 判断是否需要捕获 CUDA 图：如果 CUDA 图尚未初始化，或者当前批大小与之前不同，则需要捕获
        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            # 重置 CUDA 图
            self._cg_graph = None

            # 更新批大小和推理参数
            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            # 进行3次预热步骤，以捕获 CUDA 图
            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                # 因为 cfg != 1.0，需要复制隐藏状态
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            # 克隆输入 IDs 并存储
            self._cg_input_ids = input_ids.clone()
            # 创建一个空的 logits 张量，用于存储捕获的输出
            self._cg_logits = torch.empty_like(logits)

            # 初始化 CUDA 图
            g = torch.cuda.CUDAGraph()

            # 定义捕获区域
            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            # 捕获 CUDA 图
            with torch.cuda.graph(g):
                capture_region()
            
            # 保存捕获的 CUDA 图
            self._cg_graph = g

        else:
            # 如果不需要重新捕获，则复制输入 IDs 到缓存的输入 IDs
            self._cg_input_ids.copy_(input_ids)
        
        # 重放 CUDA 图，执行预计算的计算图
        self._cg_graph.replay()

        # 返回捕获的 logits
        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        """
        "Prefill" 模式：已经存在 `prefix_hidden_states`，我们希望附加新的嵌入，然后计算 logits。

        参数:
            prefix_hidden_states (torch.Tensor): 前缀隐藏状态，形状为 (prefix_batch, prefix_seq_len, dim)。
            input_ids (torch.Tensor): 输入的 token ID，形状为 (batch, seq_len)。
            inference_params (InferenceParams): 推理参数。
            cfg_scale (float): CFG 的缩放因子。

        返回:
            torch.Tensor: 计算得到的 logits，形状为 (batch, vocab_size)。
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            # 如果 CFG 缩放因子不为1，则将 input_ids 扩展到与 prefix_hidden_states 的批次大小相同
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        # 将前缀隐藏状态与新嵌入的输入连接起来，形状为 (batch, prefix_seq_len + seq_len, dim)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        # 计算 logits
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        """
        设置缓存，用于加速推理过程。

        参数:
            batch_size (int): 批次大小。
            max_seqlen (int): 最大序列长度。
            dtype (torch.dtype, 可选): 数据类型，默认为 bfloat16。

        返回:
            InferenceParams: 配置好的推理参数。
        """
        # 为骨干网络的每一层分配键值缓存
        key_value_memory_dict = {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.backbone.layers)
        }
        # 初始化每个样本的长度为0，形状为 (batch_size,)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32, device="cuda")
        # 返回配置好的推理参数
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        """
        准备条件嵌入。

        参数:
            cond_dict (Dict[str, Any]): 条件字典，包含条件信息。
            uncond_dict (Dict[str, Any], 可选): 无条件字典，默认为 None。如果为 None，则使用 cond_dict 中的必要键。

        返回:
            torch.Tensor: 条件嵌入张量，形状为 (2, batch, dim)。
        """
        if uncond_dict is None:
            # 如果未提供无条件字典，则从条件字典中提取必要键，生成无条件字典
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        # 将条件和无条件嵌入连接起来，形状为 (2, batch, dim)
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),  # 对条件和无条件字典应用前缀条件器，生成条件和无条件嵌入
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def _disallow_cb_not_zero_eos(self, logits):
        """
        禁止在非零位置生成结束标记（eos_token_id）。

        参数:
            logits (torch.Tensor): 输入的 logits，形状为 (batch, seq_len, vocab_size)。

        返回:
            torch.Tensor: 修改后的 logits，形状与输入相同。
        """
        # 创建一个与 logits 形状相同的零张量
        eos_bias = torch.zeros_like(logits)
        # 在 logits 中对非零位置的结束标记（eos_token_id）施加一个大的负偏置，抑制其生成
        eos_bias[:, 1:, self.eos_token_id] = -1e9
        # 返回修改后的 logits
        return logits + eos_bias

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
    ):
        """
        生成音频编码序列。

        该方法根据提供的前缀条件、前缀音频编码以及其他参数，生成新的音频编码序列。

        参数:
            prefix_conditioning (Tensor): 前缀条件张量，形状为 [batch_size, cond_seq_len, d_model]。
            audio_prefix_codes (Tensor, 可选): 前缀音频编码，形状为 [batch_size, 9, prefix_audio_seq_len]，默认为 None。
            max_new_tokens (int, 可选): 生成的最大新 token 数量，默认为 86 * 30。
            cfg_scale (float, 可选): CFG 的缩放因子，默认为2.0。
            batch_size (int, 可选): 批次大小，默认为1。
            sampling_params (Dict[str, Any], 可选): 采样参数，默认为 {'min_p': 0.1}。

        返回:
            Tensor: 生成的音频编码序列，形状为 [batch_size, 9, total_seq_len]。
        """
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        # 获取前缀音频编码的长度，如果为 None，则为0
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]

        # 定义未知 token 的值为 -1
        unknown_token = -1
        # 计算总序列长度：前缀条件长度 + 前缀音频编码长度 + 生成的新 token 数量
        seq_len = prefix_conditioning.shape[1] + prefix_audio_len + max_new_tokens

        # 设置缓存，批大小乘以2以支持 CFG，序列长度取最大值
        inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)

        # 初始化编码序列，形状为 [batch_size, 9, seq_len]，填充值为 unknown_token
        codes = torch.full((batch_size, 9, seq_len), unknown_token, device="cuda")
        if audio_prefix_codes is not None:
            # 如果存在前缀音频编码，则将其赋值到编码序列的前缀部分
            codes[..., :prefix_audio_len] = audio_prefix_codes

        # 对编码序列应用延迟模式
        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        # 获取延迟后的前缀音频编码，形状为 [batch_size, 9, prefix_audio_len + 1]
        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        # 在前缀条件下进行预填充，并计算初始 logits
        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        # 从 logits 中采样下一个 token
        next_token = sample_from_logits(logits, **sampling_params)

        # 计算偏移量
        offset = delayed_prefix_audio_codes.shape[2]
        # 获取当前帧，形状为 [batch_size, 9, 1]
        frame = delayed_codes[..., offset : offset + 1]
        # 使用下一个 token 替换未知 token
        frame.masked_scatter_(frame == unknown_token, next_token)

        # 更新前缀长度
        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        # 更新推理参数中的序列长度偏移量和每个样本的长度
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        # 逐步生成后续的 token
        for offset in trange(offset + 1, delayed_codes.shape[2]):
            # 获取输入 IDs，形状为 [batch_size, 9, 1]
            input_ids = delayed_codes[..., offset - 1 : offset]
            # 计算 logits
            logits = self._decode_one_token(input_ids, inference_params, cfg_scale)
            # 禁止在非零位置生成结束标记
            logits = self._disallow_cb_not_zero_eos(logits)
            # 从 logits 中采样下一个 token
            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
            # 如果超过8个时间步且有任何样本生成结束标记，则停止生成
            if offset > 8 and (next_token == self.eos_token_id).any():
                break
            
            # 获取当前帧，形状为 [batch_size, 9, 1]
            frame = delayed_codes[..., offset : offset + 1]
            # 使用下一个 token 替换未知 token
            frame.masked_scatter_(frame == unknown_token, next_token)
            # 更新推理参数中的序列长度偏移量和每个样本的长度
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

        # 恢复延迟模式后的编码序列
        out_codes = revert_delay_pattern(delayed_codes)
        # 将编码序列中大于等于1024的值替换为0
        out_codes.masked_fill_(out_codes >= 1024, 0)
        # 截取编码序列到实际生成的长度
        out_codes = out_codes[..., : offset - 9]

        # 重置 CUDA 图以避免缓存变化
        self._cg_graph = None

        # 返回生成的编码序列
        return out_codes
