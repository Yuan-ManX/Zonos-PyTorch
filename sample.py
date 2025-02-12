import torch
import torchaudio

from model import Zonos
from conditioning import make_cond_dict


# 使用预训练的混合模型 "Zyphra/Zonos-v0.1-hybrid"
# 从预训练模型 "Zyphra/Zonos-v0.1-transformer" 加载 Zonos 模型实例，并指定设备为 "cuda"
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")

# 加载音频文件，返回音频张量和采样率
wav, sampling_rate = torchaudio.load("audio.wav")
# 使用加载的音频生成说话人嵌入
# 调用模型的 make_speaker_embedding 方法，传入音频张量和采样率，生成说话人嵌入
speaker = model.make_speaker_embedding(wav, sampling_rate)

# 设置随机种子为 421，以确保结果的可重复性
torch.manual_seed(421)

# 构建条件字典
# 调用 make_cond_dict 函数，传入文本 "Hello, world!"，生成的说话人嵌入，以及语言代码 "en-us"
# 生成的条件字典包含文本、说话人、语言等信息，用于控制生成过程
cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
# 准备条件嵌入
# 调用模型的 prepare_conditioning 方法，传入条件字典，生成用于生成过程的条件嵌入
conditioning = model.prepare_conditioning(cond_dict)

# 生成音频编码
# 调用模型的 generate 方法，传入条件嵌入，生成音频编码
# generate 方法根据条件嵌入生成音频编码序列
codes = model.generate(conditioning)

# 解码音频编码，生成音频波形
# 调用自动编码器的 decode 方法，传入生成的音频编码，生成音频波形
# 将生成的音频波形移动到 CPU 设备，并转换为 NumPy 数组（如果需要）
wavs = model.autoencoder.decode(codes).cpu()

# 将生成的音频保存为 WAV 格式
torchaudio.save("output.wav", wavs[0], model.autoencoder.sampling_rate)
