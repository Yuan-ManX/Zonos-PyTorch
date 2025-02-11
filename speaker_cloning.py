import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download


class logFbankCal(nn.Module):
    """
    对数滤波器组能量（Log Mel-Frequency Bank）计算模块。

    该模块使用梅尔频谱图（Mel-spectrogram）计算音频的梅尔频谱特征，并进行对数变换和均值归一化。
    """
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 80,
    ):
        """
        初始化 logFbankCal 模块。

        参数:
            sample_rate (int, 可选): 音频的采样率，默认为 16,000 Hz。
            n_fft (int, 可选): FFT 的窗口大小，默认为 512。
            win_length (float, 可选): 窗长度（秒），默认为 0.025 秒（25 毫秒）。
            hop_length (float, 可选): 帧移（秒），默认为 0.01 秒（10 毫秒）。
            n_mels (int, 可选): 梅尔滤波器组数量，默认为 80。
        """
        super().__init__()
        # 初始化梅尔频谱图计算模块
        self.fbankCal = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,  # 设置采样率
            n_fft=n_fft,  # 设置 FFT 窗口大小
            win_length=int(win_length * sample_rate),  # 将窗长度从秒 转换为样本数
            hop_length=int(hop_length * sample_rate),  # 将帧移从秒 转换为样本数
            n_mels=n_mels,  # 设置梅尔滤波器组数量
        )

    def forward(self, x):
        """
        前向传播方法。

        该方法对输入音频张量进行梅尔频谱图计算、对数变换和均值归一化。

        参数:
            x (torch.Tensor): 输入的音频张量，形状为 (batch, channels, samples)。

        返回:
            torch.Tensor: 归一化后的梅尔频谱特征，形状为 (batch, n_mels, time_frames)。
        """
        # 计算梅尔频谱图，形状为 (batch, n_mels, time_frames)
        out = self.fbankCal(x)
        # 对梅尔频谱图进行对数变换，并加上一个小的常数以防止对数计算中的数值不稳定
        out = torch.log(out + 1e-6)
        # 对对数梅尔频谱图进行均值归一化
        # 计算每个时间步的均值，并从每个频率通道中减去该均值
        out = out - out.mean(axis=2).unsqueeze(dim=2)

        # 返回归一化后的梅尔频谱特征
        return out


class ASP(nn.Module):
    """
    注意力统计池化（Attentive Statistics Pooling）模块。

    该模块通过注意力机制对输入特征进行加权池化，生成均值和标准差特征。
    """
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        """
        初始化 ASP 模块。

        参数:
            in_planes (int): 输入特征维度。
            acoustic_dim (int): 声学特征维度。
        """
        super(ASP, self).__init__()
        # 计算输出映射大小
        outmap_size = int(acoustic_dim / 8)
        # 计算输出维度
        self.out_dim = in_planes * 8 * outmap_size * 2

        # 定义注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        """
        前向传播方法。

        该方法对输入特征进行注意力加权池化，生成均值和标准差特征。

        参数:
            x (torch.Tensor): 输入的隐藏状态，形状为 (batch, in_planes * 8 * outmap_size, time_steps)。

        返回:
            torch.Tensor: 池化后的特征，形状为 (batch, in_planes * 8 * outmap_size * 2)。
        """
        # 重塑输入张量形状为 (batch, in_planes * 8 * outmap_size, time_steps)
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        # 应用注意力机制，生成注意力权重，形状为 (batch, 128, time_steps)
        w = self.attention(x)
        # 计算加权均值，形状为 (batch, in_planes * 8 * outmap_size)
        mu = torch.sum(x * w, dim=2)
        # 计算加权标准差，形状为 (batch, in_planes * 8 * outmap_size)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        # 连接均值和标准差，形状为 (batch, in_planes * 8 * outmap_size * 2)
        x = torch.cat((mu, sg), 1)

        # 重塑输出张量形状为 (batch, in_planes * 8 * outmap_size * 2)
        x = x.view(x.size()[0], -1)
        # 返回池化后的特征
        return x


class SimAMBasicBlock(nn.Module):
    """
    SimAM 基础残差块（Basic Residual Block）类。

    该类实现了带有 SimAM（Simple Attention Module）机制的残差块，用于构建更深的神经网络。
    """
    # 残差块输出通道数相对于输入通道数的扩展倍数，默认为1
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        """
        初始化 SimAMBasicBlock。

        参数:
            ConvLayer (nn.Module): 卷积层类，例如 nn.Conv2d。
            NormLayer (nn.Module): 归一化层类，例如 nn.BatchNorm2d。
            in_planes (int): 输入通道数。
            planes (int): 输出通道数。
            stride (int, 可选): 卷积步幅，默认为1。
            block_id (int, 可选): 块的编号，默认为1。
        """
        super(SimAMBasicBlock, self).__init__()
        # 定义第一个卷积层和归一化层
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 初始化归一化层，参数为输出通道数
        self.bn1 = NormLayer(planes)
        # 定义第二个卷积层和归一化层
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始化归一化层，参数为输出通道数
        self.bn2 = NormLayer(planes)
        # 定义激活函数和激活函数
        # 定义 ReLU 激活函数，并启用 in-place 操作以节省内存
        self.relu = nn.ReLU(inplace=True)
        # 定义 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义下采样层，如果需要下采样，则使用卷积层和归一化层进行下采样
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),  # 初始化归一化层，参数为扩展后的通道数
            )

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, in_planes, height, width)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch, planes * expansion, new_height, new_width)。
        """
        # 第一层卷积和归一化，然后应用 ReLU 激活函数
        out = self.relu(self.bn1(self.conv1(x)))
        # 第二层卷积和归一化
        out = self.bn2(self.conv2(out))
        # 应用 SimAM 机制
        out = self.SimAM(out)
        # 添加残差连接
        out += self.downsample(x)
        # 应用 ReLU 激活函数
        out = self.relu(out)
        # 返回输出张量
        return out

    def SimAM(self, X, lambda_p=1e-4):
        """
        SimAM 注意力机制。

        该方法通过计算每个像素的重要性权重，并将其应用于输入张量。

        参数:
            X (torch.Tensor): 输入张量，形状为 (batch, channels, height, width)。
            lambda_p (float, 可选): 正则化参数，默认为1e-4。

        返回:
            torch.Tensor: 应用 SimAM 后的张量，形状与输入相同。
        """
        # 计算像素数量减1
        n = X.shape[2] * X.shape[3] - 1
        # 计算每个像素与通道均值的差的平方
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算每个像素的方差
        v = d.sum(dim=[2, 3], keepdim=True) / n
        # 计算每个像素的重要性权重
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        # 应用 Sigmoid 函数并乘以输入张量，得到输出
        return X * self.sigmoid(E_inv)


class BasicBlock(nn.Module):
    """
    基础残差块（Basic Residual Block）类。

    该类实现了基本的残差块，用于构建更深的神经网络。
    """
    # 残差块输出通道数相对于输入通道数的扩展倍数，默认为1
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        """
        初始化 BasicBlock。

        参数:
            ConvLayer (nn.Module): 卷积层类，例如 nn.Conv2d。
            NormLayer (nn.Module): 归一化层类，例如 nn.BatchNorm2d。
            in_planes (int): 输入通道数。
            planes (int): 输出通道数。
            stride (int, 可选): 卷积步幅，默认为1。
            block_id (int, 可选): 块的编号，默认为1。
        """
        super(BasicBlock, self).__init__()
        # 定义第一个卷积层和归一化层
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 初始化归一化层，参数为输出通道数
        self.bn1 = NormLayer(planes)
        # 定义第二个卷积层和归一化层
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始化归一化层，参数为输出通道数
        self.bn2 = NormLayer(planes)
        # 定义 ReLU 激活函数，并启用 in-place 操作以节省内存
        self.relu = nn.ReLU(inplace=True)

        # 定义下采样层，如果需要下采样，则使用卷积层和归一化层进行下采样
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),  # 初始化归一化层，参数为扩展后的通道数
            )

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, in_planes, height, width)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch, planes * expansion, new_height, new_width)。
        """
        # 第一层卷积和归一化，然后应用 ReLU 激活函数
        out = self.relu(self.bn1(self.conv1(x)))
        # 第二层卷积和归一化
        out = self.bn2(self.conv2(out))
        # 添加残差连接
        out += self.downsample(x)
        # 应用 ReLU 激活函数
        out = self.relu(out)
        # 返回输出张量
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck 残差块类。

    该类实现了带有瓶颈结构的残差块，通常用于更深层次的网络架构，如 ResNet。
    """
    # 残差块输出通道数相对于输入通道数的扩展倍数，默认为4
    expansion = 4

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        """
        初始化 Bottleneck 残差块。

        参数:
            ConvLayer (nn.Module): 卷积层类，例如 nn.Conv2d。
            NormLayer (nn.Module): 归一化层类，例如 nn.BatchNorm2d。
            in_planes (int): 输入通道数。
            planes (int): 瓶颈层的通道数。
            stride (int, 可选): 卷积步幅，默认为1。
            block_id (int, 可选): 块的编号，默认为1。
        """
        super(Bottleneck, self).__init__()
        # 定义第一个1x1卷积层和批归一化层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # 初始化批归一化层，参数为输出通道数
        self.bn1 = nn.BatchNorm2d(planes)
        # 定义第二个3x3卷积层和批归一化层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 初始化批归一化层，参数为输出通道数
        self.bn2 = nn.BatchNorm2d(planes)
        # 定义第三个1x1卷积层和批归一化层
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # 初始化批归一化层，参数为扩展后的通道数
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # 定义快捷连接，如果需要下采样，则使用1x1卷积层和批归一化层进行下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),  # 初始化批归一化层，参数为扩展后的通道数
            )

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, in_planes, height, width)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch, expansion * planes, new_height, new_width)。
        """
        # 第一层1x1卷积和批归一化，然后应用 ReLU 激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二层3x3卷积和批归一化，然后应用 ReLU 激活函数
        out = F.relu(self.bn2(self.conv2(out)))
        # 第三层1x1卷积和批归一化
        out = self.bn3(self.conv3(out))
        # 添加快捷连接
        out += self.shortcut(x)
        # 应用 ReLU 激活函数
        out = F.relu(out)
        # 返回输出张量
        return out


class ResNet(nn.Module):
    """
    ResNet 模型类。

    该类实现了 ResNet 模型，支持1D, 2D, 3D 卷积和批归一化。
    """
    def __init__(self, in_planes, block, num_blocks, in_ch=1, feat_dim="2d", **kwargs):
        """
        初始化 ResNet 模型。

        参数:
            in_planes (int): 输入通道数。
            block (nn.Module): 残差块类，例如 Bottleneck 或 BasicBlock。
            num_blocks (List[int]): 每个阶段的残差块数量列表。
            in_ch (int, 可选): 输入通道数，默认为1。
            feat_dim (str, 可选): 特征维度，默认为 "2d"。
            **kwargs: 其他关键字参数。
        """
        super(ResNet, self).__init__()

        # 根据特征维度选择归一化层和卷积层
        if feat_dim == "1d":
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        elif feat_dim == "2d":
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d
        elif feat_dim == "3d":
            self.NormLayer = nn.BatchNorm3d
            self.ConvLayer = nn.Conv3d
        else:
            print("error")

        # 保存输入通道数
        self.in_planes = in_planes

        # 定义第一层卷积层和批归一化层
        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始化批归一化层，参数为输出通道数
        self.bn1 = self.NormLayer(in_planes)
        # 定义 ReLU 激活函数，并启用 in-place 操作以节省内存
        self.relu = nn.ReLU(inplace=True)
        # 定义四个阶段的残差层
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, block_id=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, block_id=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, block_id=3)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, block_id=4)

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        """
        构建一个阶段的残差层。

        参数:
            block (nn.Module): 残差块类。
            planes (int): 输出通道数。
            num_blocks (int): 该阶段的残差块数量。
            stride (int): 卷积步幅。
            block_id (int, 可选): 块的编号，默认为1。

        返回:
            nn.Sequential: 包含多个残差块的序列。
        """
        # 第一个步幅为指定步幅，其余为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 初始化层列表
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride, block_id))
            # 更新输入通道数
            self.in_planes = planes * block.expansion
        # 返回包含多个残差块的序列
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (Tensor): 输入张量，形状为 (batch, in_ch, height, width)。

        返回:
            Tensor: 输出张量，形状为 (batch, in_planes * 8 * block.expansion, new_height, new_width)。
        """
        # 第一层卷积和批归一化，然后应用 ReLU 激活函数
        x = self.relu(self.bn1(self.conv1(x)))
        # 第一个阶段的残差层
        x = self.layer1(x)
        # 第二个阶段的残差层
        x = self.layer2(x)
        # 第三个阶段的残差层
        x = self.layer3(x)
        # 第四个阶段的残差层
        x = self.layer4(x)
        # 返回输出张量
        return x


def ResNet293(in_planes: int, **kwargs):
    """
    创建 ResNet293 模型。

    参数:
        in_planes (int): 输入通道数。
        **kwargs: 其他关键字参数。

        返回:
            ResNet: 创建的 ResNet293 模型实例。
    """
    return ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs)


class ResNet293_based(nn.Module):
    """
    基于 ResNet293 的模型类。

    该类在 ResNet293 的基础上，添加了特征计算、池化、全连接层和 Dropout 层，用于构建更复杂的模型。
    """
    def __init__(
        self,
        in_planes: int = 64,
        embd_dim: int = 256,
        acoustic_dim: int = 80,
        featCal=None,
        dropout: float = 0,
        **kwargs,
    ):
        """
        初始化 ResNet293_based 模型。

        参数:
            in_planes (int, 可选): 输入通道数，默认为64。
            embd_dim (int, 可选): 嵌入维度，默认为256。
            acoustic_dim (int, 可选): 声学特征维度，默认为80。
            featCal (Optional[Any], 可选): 特征计算模块，默认为 None。
            dropout (float, 可选): Dropout 概率，默认为0（不启用 Dropout）。
            **kwargs: 其他关键字参数。
        """
        super(ResNet293_based, self).__init__()
        # 保存特征计算模块
        self.featCal = featCal
        # 初始化 ResNet293 前端网络
        self.front = ResNet293(in_planes)
        # 获取 ResNet293 中残差块的扩展倍数
        block_expansion = SimAMBasicBlock.expansion
        # 初始化 ASP（注意力统计池化）层
        self.pooling = ASP(in_planes * block_expansion, acoustic_dim)
        # 初始化全连接层，将 ASP 层的输出维度映射到嵌入维度
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        # 初始化 Dropout 层，如果 dropout 概率大于0，则启用 Dropout
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出嵌入，形状为 (batch, embd_dim)。
        """
        # 如果存在特征计算模块，则对输入进行特征计算
        x = self.featCal(x)
        # 将输入张量扩展一个维度，并传递给前端网络
        x = self.front(x.unsqueeze(dim=1))
        # 对输出进行池化
        x = self.pooling(x)

        # 如果启用了 Dropout，则应用 Dropout
        if self.drop:
            x = self.drop(x)

        # 通过瓶颈层生成嵌入
        x = self.bottleneck(x)
        # 返回嵌入
        return x


class SEModule(nn.Module):
    """
    SE（Squeeze-and-Excitation）模块类。

    该模块通过自适应地重新校准通道特征响应，增强有用特征并抑制无用特征。
    """
    def __init__(self, channels, bottleneck=128):
        """
        初始化 SE 模块。

        参数:
            channels (int): 输入和输出的通道数。
            bottleneck (int, 可选): 瓶颈层的通道数，默认为128。
        """
        super(SEModule, self).__init__()
        # 定义 SE 模块的序列结构
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 自适应平均池化，输出尺寸为1
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),  # 1x1 卷积层，用于降维
            nn.ReLU(),  # ReLU 激活函数
            # nn.BatchNorm1d(bottleneck), # 批归一化层，已注释掉
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),  # 1x1 卷积层，用于升维
            nn.Sigmoid(),  # Sigmoid 激活函数，用于生成通道权重
        )

    def forward(self, input):
        """
        前向传播方法。

        参数:
            input (torch.Tensor): 输入张量，形状为 (batch, channels, seq_len)。

        返回:
            torch.Tensor: 调整后的张量，形状与输入相同。
        """
        # 应用 SE 模块，生成通道权重
        x = self.se(input)
        # 将输入与通道权重相乘，实现通道重标定
        return input * x


class Bottle2neck(nn.Module):
    """
    Bottle2neck 模块类。

    该模块是瓶颈层的一种变体，通过扩展通道数并使用分组卷积来减少计算量，同时保持模型的表达能力。
    """
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        """
        初始化 Bottle2neck 模块。

        参数:
            inplanes (int): 输入通道数。
            planes (int): 输出通道数。
            kernel_size (int, 可选): 卷积核大小，默认为 None。
            dilation (int, 可选): 膨胀率，默认为 None。
            scale (int, 可选): 扩展比例，默认为8。
        """
        super(Bottle2neck, self).__init__()
        # 计算每个扩展通道的宽度
        width = int(math.floor(planes / scale))
        # 初始化第一个1x1卷积层，用于扩展通道数
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        # 初始化批归一化层
        self.bn1 = nn.BatchNorm1d(width * scale)
        # 计算需要多少个卷积层
        self.nums = scale - 1
        # 初始化卷积层列表
        convs = []
        # 初始化批归一化层列表
        bns = []
        # 计算填充大小
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            # 初始化卷积层，每个卷积层的输入和输出通道数均为 `width`
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            # 初始化批归一化层
            bns.append(nn.BatchNorm1d(width))

        # 使用 ModuleList 存储卷积层和批归一化层
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # 初始化最后一个1x1卷积层，用于缩小通道数
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        # 初始化批归一化层
        self.bn3 = nn.BatchNorm1d(planes)

        # 初始化 ReLU 激活函数
        self.relu = nn.ReLU()
        # 保存每个扩展通道的宽度
        self.width = width
        # 初始化 SE 模块
        self.se = SEModule(planes)

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (Tensor): 输入张量，形状为 (batch, inplanes, seq_len)。

        返回:
            Tensor: 输出张量，形状为 (batch, planes, seq_len)。
        """
        # 保存输入作为残差
        residual = x
        # 通过第一个1x1卷积层
        out = self.conv1(x)
        # 应用 ReLU 激活函数
        out = self.relu(out)
        # 应用批归一化
        out = self.bn1(out)

        # 将输出拆分为多个部分，每个部分宽度为 `width`
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                # 获取第一个部分
                sp = spx[i]
            else:
                # 将当前部分与第一个部分相加
                sp = sp + spx[i]

            # 通过卷积层
            sp = self.convs[i](sp)
            # 应用 ReLU 激活函数
            sp = self.relu(sp)
            # 应用批归一化
            sp = self.bns[i](sp)
            if i == 0:
                # 如果是第一个部分，则直接赋值
                out = sp
            else:
                # 否则，将当前部分与输出连接起来
                out = torch.cat((out, sp), 1)
        # 将最后一个部分连接到输出上
        out = torch.cat((out, spx[self.nums]), 1)

        # 通过最后一个1x1卷积层
        out = self.conv3(out)
        # 应用 ReLU 激活函数
        out = self.relu(out)
        # 应用批归一化
        out = self.bn3(out)

        # 应用 SE 模块
        out = self.se(out)
        # 添加残差连接
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN 模型类。

    该类实现了 ECAPA-TDNN 模型，用于语音识别和说话人识别等任务。该模型结合了时延神经网络（TDNN）和注意力机制，能够有效地捕捉音频信号中的时序和空间特征。
    """
    def __init__(self, C, featCal):
        """
        初始化 ECAPA-TDNN 模型。

        参数:
            C (int): 基础通道数，用于控制模型复杂度。
            featCal (Any): 特征计算模块，用于对输入音频进行预处理。
        """
        super(ECAPA_TDNN, self).__init__()
        # 保存特征计算模块
        self.featCal = featCal
        # 第一个1D 卷积层，用于初步特征提取
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        # 初始化批归一化层，参数为输出通道数 C
        self.bn1 = nn.BatchNorm1d(C)

        # 初始化 Bottle2neck 层，用于进一步的特征提取和通道扩展
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        # 初始化最后一个卷积层，将通道数从 3*C 扩展到 1536
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        # 初始化注意力机制，用于对全局特征进行加权
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # Added
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        # 初始化批归一化层，参数为3072
        self.bn5 = nn.BatchNorm1d(3072)
        # 初始化全连接层，输入维度为3072，输出维度为192
        self.fc6 = nn.Linear(3072, 192)
        # 初始化批归一化层，参数为192
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (Tensor): 输入张量，形状为 (batch, 80, seq_len)。

        返回:
            Tensor: 输出张量，形状为 (batch, 192)。
        """
        # 应用特征计算模块，对输入音频进行预处理
        x = self.featCal(x)
        # 通过第一个1D 卷积层，初步特征提取
        x = self.conv1(x)
        x = self.relu(x)
        # 应用批归一化
        x = self.bn1(x)

        # 通过三个 Bottle2neck 层进行进一步的特征提取和通道扩展
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        # 将三个 Bottle2neck 层的输出连接起来，形状为 (batch, 3*C, seq_len)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # 获取时间步长度
        t = x.size()[-1]

        # 生成全局特征，包括原始特征、均值和标准差
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )

        # 应用注意力机制，生成注意力权重
        w = self.attention(global_x)

        # 计算加权均值和加权标准差
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        # 连接均值和标准差，形状为 (batch, 3072)
        x = torch.cat((mu, sg), 1)
        # 应用批归一化
        x = self.bn5(x)
        # 通过全连接层，输出维度为192
        x = self.fc6(x)
        # 应用批归一化
        x = self.bn6(x)

        # 返回输出
        return x


class SpeakerEmbedding(nn.Module):
    """
    说话人嵌入模型类。

    该类使用预训练的 ResNet293_based 模型生成说话人嵌入。
    """
    def __init__(self, ckpt_path: str = "ResNet293_SimAM_ASP_base.pt", device: str = "cuda"):
        """
        初始化 SpeakerEmbedding 模型。

        参数:
            ckpt_path (str, 可选): 预训练模型的检查点路径，默认为 "ResNet293_SimAM_ASP_base.pt"。
            device (str, 可选): 设备类型，默认为 "cuda"。
        """
        super().__init__()
        self.device = device
        with torch.device(device):
            # 初始化 ResNet293_based 模型
            self.model = ResNet293_based()
            # 加载预训练模型的权重，忽略不匹配的键，并使用内存映射以节省内存
            self.model.load_state_dict(torch.load(ckpt_path, weights_only=True, mmap=True))
            # 初始化特征计算模块为 logFbankCal
            self.model.featCal = logFbankCal()

        # 冻结模型参数，并设置为评估模式
        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        """
        获取模型参数的数据类型。

        返回:
            torch.dtype: 模型参数的数据类型。
        """
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        """
        获取音频重采样器。

        参数:
            orig_sample_rate (int): 原始音频的采样率。

        返回:
            torchaudio.transforms.Resample: 音频重采样器。
        """
        # 返回重采样器，将音频重采样到 16,000 Hz
        return torchaudio.transforms.Resample(orig_sample_rate, 16_000).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        准备输入音频。

        参数:
            wav (Tensor): 输入的音频张量，形状为 (channels, samples) 或 (1, samples)。
            sample_rate (int): 输入音频的采样率。

        返回:
            Tensor: 预处理后的音频张量，形状为 (1, samples)。
        """
        assert wav.ndim < 3
        if wav.ndim == 2:
            # 如果输入张量的维度为2，则在第0维上添加一个维度，形状变为 (1, samples)
            wav = wav.mean(0, keepdim=True)
        # 对音频进行重采样
        wav = self._get_resampler(sample_rate)(wav)
        # 返回预处理后的音频张量
        return wav

    def forward(self, wav: torch.Tensor, sample_rate: int):
        """
        前向传播方法。

        参数:
            wav (Tensor): 输入的音频张量，形状为 (channels, samples) 或 (1, samples)。
            sample_rate (int): 输入音频的采样率。

        返回:
            Tensor: 生成的说话人嵌入，形状为 (1, embedding_dim)。
        """
        # 准备输入音频
        wav = self.prepare_input(wav, sample_rate).to(self.device, self.dtype)
        # 通过模型生成说话人嵌入，并将其移动到原始设备
        return self.model(wav).to(wav.device)


class SpeakerEmbeddingLDA(nn.Module):
    """
    基于 LDA 的说话人嵌入模型类。

    该类使用预训练的 ResNet293_based 模型生成说话人嵌入，并使用线性判别分析（LDA）进行进一步处理。
    """
    def __init__(
        self,
        device: str = "cuda",
    ):
        """
        初始化 SpeakerEmbeddingLDA 模型。

        参数:
            device (str, 可选): 设备类型，默认为 "cuda"。
        """
        super().__init__()
        # 从 Hugging Face Hub 下载预训练模型的检查点路径
        spk_model_path = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding", filename="ResNet293_SimAM_ASP_base.pt")
        lda_spk_model_path = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding", filename="ResNet293_SimAM_ASP_base_LDA-128.pt")

        self.device = device
        with torch.device(device):
            # 初始化 SpeakerEmbedding 模型
            self.model = SpeakerEmbedding(spk_model_path, device)
            # 加载 LDA 模型的权重
            lda_sd = torch.load(lda_spk_model_path, weights_only=True)
            # 获取 LDA 模型的输入和输出特征维度
            out_features, in_features = lda_sd["weight"].shape
            # 初始化线性判别分析层
            self.lda = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
            # 加载 LDA 模型的权重
            self.lda.load_state_dict(lda_sd)

        # 冻结模型参数，并设置为评估模式
        self.requires_grad_(False).eval()

    def forward(self, wav: torch.Tensor, sample_rate: int):
        """
        前向传播方法。

        参数:
            wav (Tensor): 输入的音频张量，形状为 (channels, samples) 或 (1, samples)。
            sample_rate (int): 输入音频的采样率。

        返回:
            Tuple[Tensor, Tensor]: 包含原始嵌入和 LDA 处理后的嵌入的元组。
        """
        # 生成原始的说话人嵌入
        emb = self.model(wav, sample_rate).to(torch.float32)
        # 通过 LDA 层处理嵌入
        return emb, self.lda(emb)
