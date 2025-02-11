from functools import cache
from typing import Any, Literal, Iterable
import torch
import torch.nn as nn

from config import PrefixConditionerConfig


class Conditioner(nn.Module):
    """
    Conditioner 类。

    该类用于处理条件信息，根据不同的配置对输入条件进行投影，并生成条件嵌入。
    """
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int | None = None,
        projection: Literal["none", "linear", "mlp"] = "none",
        uncond_type: Literal["learned", "none"] = "none",
        **kwargs,
    ):
        """
        初始化 Conditioner。

        参数:
            output_dim (int): 输出维度。
            name (str): 条件器的名称，用于标识不同的条件器。
            cond_dim (int, 可选): 条件维度，如果为 None，则默认为 output_dim。
            projection (Literal["none", "linear", "mlp"], 可选): 投影方式，默认为 "none"。
            uncond_type (Literal["learned", "none"], 可选): 无条件类型，默认为 "none"。
            **kwargs: 其他关键字参数。
        """
        super().__init__()
        # 保存条件器的名称
        self.name = name
        # 保存输出维度
        self.output_dim = output_dim
        # 设置条件维度，如果未提供，则默认为输出维度
        self.cond_dim = cond_dim = cond_dim or output_dim

        # 根据投影方式初始化投影层
        if projection == "linear":
            # 线性投影层，将条件维度映射到输出维度
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim), # 线性层
                nn.SiLU(), # SiLU 激活函数
                nn.Linear(output_dim, output_dim), # 另一个线性层
            )
        else:
            # 无投影层，直接返回输入
            self.project = nn.Identity()

        # 初始化无条件向量
        self.uncond_vector = None
        if uncond_type == "learned":
            # 初始化无条件向量为全零张量，并设置为可学习的参数
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        """
        应用条件。

        该方法需要子类实现，用于根据输入条件生成条件嵌入。

        参数:
            *inputs (Any): 输入条件，可以是任意类型。

        返回:
            Tensor: 生成的条件嵌入。

        异常:
            NotImplementedError: 如果子类未实现此方法，则抛出 NotImplementedError。
        """
        raise NotImplementedError()

    def forward(self, inputs: tuple[Any, ...] | None) -> torch.Tensor:
        """
        前向传播方法。

        该方法根据输入条件生成条件嵌入，并根据配置应用投影和生成无条件向量。

        参数:
            inputs (Tuple[Any, ...], 可选): 输入条件，默认为 None。

        返回:
            Tensor: 生成的条件嵌入或无条件向量。
        """
        if inputs is None:
            # 如果输入为 None，则返回无条件向量
            assert self.uncond_vector is not None
             # 返回无条件向量，并调整形状为 (1, 1, output_dim)
            return self.uncond_vector.data.view(1, 1, -1)

        # 应用条件，生成条件嵌入
        cond = self.apply_cond(*inputs)
        # 对条件嵌入进行投影
        cond = self.project(cond)
        # 返回最终的条件嵌入
        return cond


###################################### ESPEAK CONTAINMENT ZONE ######################################
import re
import unicodedata

import inflect
import torch
import torch.nn as nn
from kanjize import number2kanji
from phonemizer.backend import EspeakBackend
from sudachipy import Dictionary, SplitMode


###################################### Number normalization ######################################

# 初始化 inflect 引擎，用于数字到单词的转换
_inflect = inflect.engine()

# 定义正则表达式模式，用于匹配不同类型的数字格式
# 匹配包含逗号的数字，例如 "1,000"
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
# 匹配包含小数点的数字，例如 "3.14"
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
# 匹配以英镑符号 £ 开头的数字，例如 "£1,234.56"
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
# 匹配以美元符号 $ 开头的数字，例如 "\$1,234.56"
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
# 匹配序数词，例如 "1st", "2nd", "3rd", "4th"
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
# 匹配纯数字，例如 "1234"
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    """
    移除数字中的逗号。

    参数:
        m (re.Match): 正则表达式匹配对象。

    返回:
        str: 移除逗号后的数字字符串。
    """
    # 替换逗号为""，即移除逗号
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    """
    将小数点展开为 " point "。

    参数:
        m (re.Match): 正则表达式匹配对象。

    返回:
        str: 展开小数点后的字符串。
    """
    # 将小数点替换为 " point "
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    """
    将$展开为文字描述。

    参数:
        m (re.Match): 正则表达式匹配对象。

    返回:
        str: 展开后的美元金额字符串。
    """
    # 获取匹配的数字部分，例如 "\$1,234.56" 中的 "1,234.56"
    match = m.group(1)
    # 按小数点分割
    parts = match.split(".")
    
    if len(parts) > 2:
        # 如果小数点超过一个，则返回原始字符串加上 " dollars"
        return match + " dollars"  # Unexpected format
    
    # 获取dollars部分，如果为空则设为0
    dollars = int(parts[0]) if parts[0] else 0
    # 获取dollars部分，如果为空则设为0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0

    if dollars and cents:
        # 如果既有dollar又有cent，则分别转换为单词并连接
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        # 如果只有dollars，则转换为单词
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        # 如果只有cents，则转换为单词
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        # 如果没有dollars和cents，则返回 "zero dollars"
        return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    """
    将序数词转换为文字描述。

    参数:
        m (re.Match): 正则表达式匹配对象。

    返回:
        str: 展开后的序数词字符串。
    """
    # 使用 inflect 引擎将序数词转换为文字
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    """
    将数字转换为文字描述。

    参数:
        m (re.Match): 正则表达式匹配对象。

    返回:
        str: 展开后的数字字符串。
    """
    # 将匹配的数字字符串转换为整数
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"  # 特殊情况处理
        elif num > 2000 and num < 2010:
            # 处理2000到2009之间的数字
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            # 处理整百的数字
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            # 处理其他数字
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        # 使用 inflect 引擎将数字转换为文字
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    """
    规范化文本中的数字。

    该函数将文本中的不同格式的数字转换为文字描述。

    参数:
        text (str): 输入的文本字符串。

    返回:
        str: 规范化后的文本字符串。
    """
    # 依次应用不同的正则表达式替换规则
    text = re.sub(_comma_number_re, _remove_commas, text)  # 移除数字中的逗号
    text = re.sub(_pounds_re, r"\1 pounds", text)  # 在英镑符号后的数字后添加 " pounds"
    text = re.sub(_dollars_re, _expand_dollars, text)  # 将dollars展开为文字描述
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)  # 将小数点展开为 " point "
    text = re.sub(_ordinal_re, _expand_ordinal, text)  # 将序数词转换为文字描述
    text = re.sub(_number_re, _expand_number, text)  # 将数字转换为文字描述
    # 返回规范化后的文本
    return text


# 定义特殊标记的 ID
# PAD: 填充标记; UNK: 未知词标记; BOS: 句子开始标记; EOS: 句子结束标记
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3  
# 特殊标记 ID 列表
SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

# 定义标点符号和字母（包括 IPA 国际音标）
# 标点符号
_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'
# 英文字母
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# IPA 国际音标
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

# 合并标点符号、字母和 IPA 国际音标，形成完整的符号列表
symbols = [*_punctuation, *_letters, *_letters_ipa]
# 创建符号到 ID 的映射字典，ID 从 len(SPECIAL_TOKEN_IDS) 开始编号
_symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}


# 定义辅助函数，用于获取符号的 ID
def _get_symbol_id(s: str) -> int:
    """
    获取符号的 ID。

    参数:
        s (str): 输入的符号。

    返回:
        int: 符号对应的 ID。如果符号不存在于符号表中，则返回 UNK_ID（1）。
    """
    # 如果符号不存在于符号表中，则返回 UNK_ID
    return _symbol_to_id.get(s, 1)


# 定义函数，用于将文本转换为符号 ID 列表
def get_symbol_ids(text: str) -> list[int]:
    """
    将文本转换为符号 ID 列表。

    参数:
        text (str): 输入的文本。

    返回:
        List[int]: 符号 ID 列表。
    """
    # 对文本中的每个字符应用 _get_symbol_id 函数，返回 ID 列表
    return list(map(_get_symbol_id, text))


# 定义函数，用于将音素列表转换为张量和长度列表
def tokenize_phonemes(phonemes: list[str]) -> tuple[torch.Tensor, list[int]]:
    """
    将音素列表转换为张量和长度列表。

    参数:
        phonemes (List[str]): 输入的音素列表。

    返回:
        Tuple[torch.Tensor, List[int]]: 音素 ID 的张量和每个音素序列的长度列表。
    """
    # 为每个音素序列添加 BOS 和 EOS 标记，并转换为 ID 列表
    phoneme_ids = [[BOS_ID, *get_symbol_ids(phonemes), EOS_ID] for phonemes in phonemes]
    # 获取每个音素序列的长度
    lengths = list(map(len, phoneme_ids))
    # 找到最长的音素序列长度
    longest = max(lengths)
    # 对所有音素序列进行填充，使其长度相同
    phoneme_ids = [[PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
    # 将音素 ID 列表转换为张量，并返回张量和长度列表
    return torch.tensor(phoneme_ids), lengths


# 定义函数，用于规范化日语文本
def normalize_jp_text(text: str, tokenizer=Dictionary(dict="full").create()) -> str:
    """
    规范化日语文本。

    参数:
        text (str): 输入的日语文本。
        tokenizer: 分词器，默认为使用 "full" 字典的 Separator。

    返回:
        str: 规范化后的日语文本。
    """
    # 规范化 Unicode 字符
    text = unicodedata.normalize("NFKC", text)
    # 将数字转换为汉字数字
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    # 使用分词器对文本进行分词，并获取每个词的读音形式
    final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
    return final_text


# 定义函数，用于清理文本列表
def clean(texts: list[str], languages: list[str]) -> list[str]:
    """
    清理文本列表。

    对每个文本，根据其语言应用不同的规范化方法。

    参数:
        texts (List[str]): 输入的文本列表。
        languages (List[str]): 对应文本的语言列表。

    返回:
        List[str]: 清理后的文本列表。
    """
    texts_out = []
    for text, language in zip(texts, languages):
        if "ja" in language:
            # 如果语言包含 "ja"，则规范化日语文本
            text = normalize_jp_text(text)
        else:
            # 否则，规范化数字
            text = normalize_numbers(text)
        texts_out.append(text)
    return texts_out


# 定义缓存函数，用于获取语音合成后端
@cache
def get_backend(language: str) -> "EspeakBackend":
    """
    获取语音合成后端。

    参数:
        language (str): 语言代码。

    返回:
        EspeakBackend: 语音合成后端实例。
    """
    import logging

    from phonemizer.backend import EspeakBackend

    # 获取 phonemizer 的日志记录器
    logger = logging.getLogger("phonemizer")

    # 初始化 EspeakBackend，启用标点符号和重音，并使用自定义的标点符号列表
    backend = EspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=_punctuation,
        logger=logger,
    )

    # 设置日志级别为 ERROR
    logger.setLevel(logging.ERROR)
    return backend


# 定义函数，用于将文本转换为音素
def phonemize(texts: list[str], languages: list[str]) -> list[str]:
    """
    将文本转换为音素。

    参数:
        texts (List[str]): 输入的文本列表。
        languages (List[str]): 对应文本的语言列表。

    返回:
        List[str]: 音素列表。
    """
    # 清理文本
    texts = clean(texts, languages)

    batch_phonemes = []
    for text, language in zip(texts, languages):
        # 获取语音合成后端
        backend = get_backend(language)
        # 将文本转换为音素，并去除多余的空格
        phonemes = backend.phonemize([text], strip=True)
        # 添加音素到批次列表中
        batch_phonemes.append(phonemes[0])

    # 返回音素列表
    return batch_phonemes


# 定义 EspeakPhonemeConditioner 类，继承自 Conditioner 类
class EspeakPhonemeConditioner(Conditioner):
    """
    EspeakPhonemeConditioner 类。

    该类使用 Espeak 后端将文本转换为音素，并嵌入音素以供模型使用。
    """
    def __init__(self, output_dim: int, **kwargs):
        """
        初始化 EspeakPhonemeConditioner。

        参数:
            output_dim (int): 输出维度。
            **kwargs: 其他关键字参数。
        """
        super().__init__(output_dim, **kwargs)
        # 初始化音素嵌入层，嵌入层的尺寸为 (总符号数, 输出维度)
        self.phoneme_embedder = nn.Embedding(len(SPECIAL_TOKEN_IDS) + len(symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        """
        应用条件，将文本转换为音素并嵌入。

        参数:
            texts (List[str]): 输入的文本列表。
            languages (List[str]): 对应文本的语言列表。

        返回:
            torch.Tensor: 嵌入后的音素张量。
        """
        device = self.phoneme_embedder.weight.device

        # 将文本转换为音素
        phonemes = phonemize(texts, languages)
        # 将音素转换为 ID 列表
        phoneme_ids, _ = tokenize_phonemes(phonemes)
        # 嵌入音素 ID
        phoneme_embeds = self.phoneme_embedder(phoneme_ids.to(device))

        # 返回嵌入后的音素张量
        return phoneme_embeds


###################################### ESPEAK CONTAINMENT ZONE ######################################

class FourierConditioner(Conditioner):
    """
    FourierConditioner 类。

    该类使用傅里叶变换将输入数据转换为傅里叶域表示，并生成条件嵌入。
    """
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs,
    ):
        """
        初始化 FourierConditioner。

        参数:
            output_dim (int): 输出维度，必须是偶数。
            input_dim (int, 可选): 输入维度，默认为1。
            std (float, 可选): 权重初始化的标准差，默认为1.0。
            min_val (float, 可选): 输入的最小值，默认为0.0。
            max_val (float, 可选): 输入的最大值，默认为1.0。
            **kwargs: 其他关键字参数。
        """
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)

        # 初始化权重，形状为 (output_dim // 2, input_dim)，并使用正态分布初始化
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        # 保存输入维度、最小值和最大值
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用条件，将输入数据转换为傅里叶域表示。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)。

        返回:
            torch.Tensor: 傅里叶域表示，形状为 (batch_size, seq_len, output_dim)。
        """
        assert x.shape[-1] == self.input_dim
        # 将输入数据归一化到 [0, 1] 范围
        x = (x - self.min_val) / (self.max_val - self.min_val)  # [batch_size, seq_len, input_dim]
        # 计算傅里叶变换，形状为 (batch_size, seq_len, output_dim // 2)
        f = 2 * torch.pi * x.to(self.weight.dtype) @ self.weight.T  # [batch_size, seq_len, output_dim // 2]
        # 将傅里叶变换结果转换为余弦和正弦表示，并连接
        return torch.cat([f.cos(), f.sin()], dim=-1)  # [batch_size, seq_len, output_dim]


# 定义 IntegerConditioner 类，继承自 Conditioner
class IntegerConditioner(Conditioner):
    """
    IntegerConditioner 类。

    该类将整数输入嵌入到高维空间中，生成条件嵌入。
    """
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        """
        初始化 IntegerConditioner。

        参数:
            output_dim (int): 输出维度。
            min_val (int, 可选): 整数的最小值，默认为0。
            max_val (int, 可选): 整数的最大值，默认为512。
            **kwargs: 其他关键字参数。
        """
        super().__init__(output_dim, **kwargs)
        # 保存最小值
        self.min_val = min_val
        # 保存最大值
        self.max_val = max_val
        # 初始化整数嵌入层，嵌入层的尺寸为 (max_val - min_val + 1, output_dim)
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用条件，将整数输入嵌入到高维空间中。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, 1)。

        返回:
            torch.Tensor: 嵌入后的张量，形状为 (batch_size, seq_len, output_dim)。
        """
        assert x.shape[-1] == 1
        # 将整数输入减去最小值，并嵌入到高维空间中
        return self.int_embedder(x.squeeze(-1) - self.min_val)  # [batch_size, seq_len, output_dim]


# 定义 PassthroughConditioner 类，继承自 Conditioner
class PassthroughConditioner(Conditioner):
    """
    PassthroughConditioner 类。

    该类将输入数据直接作为条件嵌入输出，不进行任何变换。
    """
    def __init__(self, output_dim: int, **kwargs):
        """
        初始化 PassthroughConditioner。

        参数:
            output_dim (int): 输出维度。
            **kwargs: 其他关键字参数。
        """
        super().__init__(output_dim, **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用条件，将输入数据直接作为输出。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, cond_dim)。

        返回:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        assert x.shape[-1] == self.cond_dim
        # 直接返回输入张量
        return x


# 定义条件器类映射字典，将类名映射到具体的类
_cond_cls_map = {
    "PassthroughConditioner": PassthroughConditioner,  # 无变换条件器
    "EspeakPhonemeConditioner": EspeakPhonemeConditioner,  # Espeak 音素条件器
    "FourierConditioner": FourierConditioner,  # 傅里叶条件器
    "IntegerConditioner": IntegerConditioner,  # 整数嵌入条件器
}


# 定义函数，用于根据配置列表构建条件器列表
def build_conditioners(conditioners: list[dict], output_dim: int) -> list[Conditioner]:
    """
    根据配置列表构建条件器列表。

    参数:
        conditioners (List[Dict[str, Any]]): 条件器配置列表，每个元素是一个字典，包含条件器的配置参数。
        output_dim (int): 输出维度。

        返回:
            List[Conditioner]: 构建好的条件器列表。
        """
    return [_cond_cls_map[config["type"]](output_dim, **config) for config in conditioners]


# 定义 PrefixConditioner 类，继承自 Conditioner
class PrefixConditioner(Conditioner):
    """
    PrefixConditioner 类。

    该类处理前缀条件，根据配置应用一系列条件器，并生成最终的条件嵌入。
    """
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        """
        初始化 PrefixConditioner。

        参数:
            config (PrefixConditionerConfig): 前缀条件器的配置参数。
            output_dim (int): 输出维度。
        """
        # 调用父类的初始化方法，传入条件器名称和投影方式
        super().__init__(output_dim, "prefix", projection=config.projection)
        # 根据配置构建条件器列表
        self.conditioners = nn.ModuleList(build_conditioners(config.conditioners, output_dim))
        # 初始化层归一化层
        self.norm = nn.LayerNorm(output_dim)
        # 获取所有需要无条件向量的条件器名称
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def forward(self, cond_dict: dict) -> torch.Tensor:
        """
        前向传播方法。

        该方法根据输入的条件字典，应用所有条件器，并生成最终的条件嵌入。

        参数:
            cond_dict (Dict[str, Any]): 输入的条件字典，包含各种条件信息。

        返回:
            torch.Tensor: 生成的条件嵌入，形状为 (batch, seq_len, output_dim)。
        """
        # 检查是否包含所有必需的条件键
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        
        # 初始化条件列表
        conds = []
        # 遍历所有条件器，应用条件并添加到列表中
        for conditioner in self.conditioners:
            conds.append(conditioner(cond_dict.get(conditioner.name)))

        # 获取最大批次大小
        max_bsz = max(map(len, conds))
        assert all(c.shape[0] in (max_bsz, 1) for c in conds)

        # 对条件进行扩展，以确保所有条件的批次大小一致
        conds = [c.expand(max_bsz, -1, -1) for c in conds]
        # 连接所有条件，并应用层归一化
        return self.norm(self.project(torch.cat(conds, dim=-2)))


# 定义支持的语言代码列表
supported_language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]

def make_cond_dict(
    text: str = "It would be nice to have time for testing, indeed.",
    language: str = "en-us",
    speaker: torch.Tensor | None = None,
    emotion: torch.Tensor | None = None,
    fmax: float = 22050.0,
    pitch_std: float = 20.0,
    speaking_rate: float = 15.0,
    vqscore_8: torch.Tensor | None = None,
    ctc_loss: float = 0.0,
    dnsmos_ovrl: float = 4.0,
    speaker_noised: bool = False,
    unconditional_keys: Iterable[str] = {"vqscore_8", "dnsmos_ovrl"},
    device: str = "cuda",
    speaker_dim: int = 128,
) -> dict:
    """
    A helper to build the 'cond_dict' that the model expects.
    By default, it will generate a random speaker embedding
    """
    """
    构建模型期望的 `cond_dict` 字典。

    默认情况下，它将生成一个随机的说话人嵌入。

    参数:
        text (str, 可选): 输入的文本，默认为 "It would be nice to have time for testing, indeed."。
        language (str, 可选): 语言代码，默认为 "en-us"。
        speaker (Tensor, 可选): 说话人嵌入，默认为 None。如果为 None，则生成随机的说话人嵌入。
        emotion (Tensor, 可选): 情感嵌入，默认为 None。如果为 None，则使用默认的情感嵌入。
        fmax (float, 可选): 最大频率，默认为22050.0 Hz。
        pitch_std (float, 可选): 音高标准差，默认为20.0。
        speaking_rate (float, 可选): 语速，默认为15.0。
        vqscore_8 (Tensor, 可选): VQ 评分（8维），默认为 None。如果为 None，则使用默认值。
        ctc_loss (float, 可选): CTC 损失，默认为0.0。
        dnsmos_ovrl (float, 可选): DNSMOS 总体评分，默认为4.0。
        speaker_noised (bool, 可选): 是否对说话人进行噪声处理，默认为 False。
        unconditional_keys (Iterable[str], 可选): 无条件键集合，默认为 {"vqscore_8", "dnsmos_ovrl"}。
        device (str, 可选): 设备类型，默认为 "cuda"。
        speaker_dim (int, 可选): 说话人嵌入的维度，默认为128。

    返回:
        dict: 构建好的条件字典。
    """
    assert language.lower() in supported_language_codes, "Please pick a supported language"

    # 创建语言代码到 ID 的映射字典
    language_code_to_id = {lang: i for i, lang in enumerate(supported_language_codes)}

    if speaker is None:
        # 如果未提供说话人嵌入，则生成随机的说话人嵌入
        speaker = (3.0 * torch.randn((1, 1, speaker_dim), device=device)).unsqueeze(0).to(torch.bfloat16)

    if emotion is None:
        # 如果未提供情感嵌入，则使用默认的情感嵌入
        emotion = torch.tensor(
            [[0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]],
            device=device,
        )

    if vqscore_8 is None:
        # 如果未提供 VQ 评分（8维），则使用默认值
        vqscore_8 = torch.tensor([0.78] * 8, device=device).view(1, 8)

    # 构建条件字典
    cond_dict = {
        "espeak": ([text], [language]),  # 文本和语言
        "speaker": speaker,  # 说话人嵌入
        "emotion": emotion,  # 情感嵌入
        "fmax": torch.tensor([[fmax]], device=device),  # 最大频率
        "pitch_std": torch.tensor([[pitch_std]], device=device),  # 音高标准差
        "speaking_rate": torch.tensor([[speaking_rate]], device=device),  # 语速
        "language_id": torch.tensor([language_code_to_id[language]], device=device),  # 语言 ID
        "vqscore_8": vqscore_8,  # VQ 评分（8维）
        "ctc_loss": torch.tensor([[ctc_loss]], device=device),  # CTC 损失
        "dnsmos_ovrl": torch.tensor([[dnsmos_ovrl]], device=device),  # DNSMOS 总体评分
        "speaker_noised": torch.tensor([[int(speaker_noised)]], device=device),  # 是否对说话人进行噪声处理
    }

    # 对条件字典中的非 "espeak" 和 "speaker" 的键进行扩展
    for k in cond_dict:
        if k != "espeak" and k != "speaker":
            cond_dict[k] = cond_dict[k].unsqueeze(0).unsqueeze(0)

    # 返回构建好的条件字典
    return cond_dict
