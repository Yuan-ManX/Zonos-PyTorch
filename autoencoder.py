import math
import torch
import torchaudio
from transformers.models.dac import DacModel


class DACAutoencoder:
    """
    DAC 自动编码器类。

    该类使用预训练的 DAC（Discrete Autoencoder for Audio Compression）模型对音频进行编码和解码。
    """
    def __init__(self):
        """
        初始化 DAC 自动编码器。

        该方法加载预训练的 DAC 模型，并设置模型为评估模式且不计算梯度。
        """
        super().__init__()
        # 从预训练模型加载 DacModel 实例
        self.dac = DacModel.from_pretrained("descript/dac_44khz")
        # 将模型设置为评估模式，并冻结其参数以防止梯度计算
        self.dac.eval().requires_grad_(False)

        # 获取模型的配置参数
        # 获取码本大小
        self.codebook_size = self.dac.config.codebook_size
        # 获取码本数量
        self.num_codebooks = self.dac.quantizer.n_codebooks
        # 获取采样率
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        对输入音频进行预处理。

        该方法将输入音频重采样到 44.1kHz，并进行右填充以确保音频长度是 512 的倍数。

        参数:
            wav (torch.Tensor): 输入音频张量，形状为 (batch, channels, samples)。
            sr (int): 输入音频的采样率。

        返回:
            torch.Tensor: 预处理后的音频张量，形状为 (batch, channels, padded_samples)。
        """
        # 将输入音频重采样到 44.1kHz
        wav = torchaudio.functional.resample(wav, sr, 44_100)

        # 计算需要填充的长度，以确保音频长度是 512 的倍数
        right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]

        # 对音频进行右填充，填充值为0
        return torch.nn.functional.pad(wav, (0, right_pad))

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        对输入音频进行编码。

        该方法使用 DAC 模型对预处理后的音频进行编码，生成离散音频代码。

        参数:
            wav (torch.Tensor): 输入音频张量，形状为 (batch, channels, samples)。

        返回:
            torch.Tensor: 编码后的音频代码，形状为 (batch, num_codebooks, codebook_size)。
        """
        # 对输入音频进行编码，生成音频代码
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        对编码后的音频代码进行解码。

        该方法使用 DAC 模型对编码后的音频代码进行解码，生成原始音频波形。

        参数:
            codes (torch.Tensor): 输入音频代码，形状为 (batch, num_codebooks, codebook_size)。

        返回:
            torch.Tensor: 解码后的音频波形，形状为 (batch, 1, samples)。
        """
        # 对音频代码进行解码，生成音频波形
        # 在时间维度上添加一个维度，形状为 (batch, 1, samples)
        return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1)
