"""
项目配置文件 - 保存所有项目级别的常数和配置

你可以在这里修改默认参数、日志级别等设置
"""

# ============================================================================
# 变声参数配置
# ============================================================================

# 默认参数组合
DEFAULT_PARAMS = {
    "pitch_ratio": 1.75,        # 默认音调升高倍数
    "formant_ratio": 1.15       # 默认共振峰平移倍数
}

# 预设参数组合
PRESET_PARAMS = {
    "light_female": {           # 轻微女化
        "pitch_ratio": 1.65,
        "formant_ratio": 1.12,
        "description": "轻微女化，保留部分男性特征"
    },
    "natural_female": {         # 自然女声（推荐）
        "pitch_ratio": 1.75,
        "formant_ratio": 1.15,
        "description": "自然女声，最平衡的效果 ⭐"
    },
    "young_female": {           # 高亢少女
        "pitch_ratio": 1.90,
        "formant_ratio": 1.18,
        "description": "高亢女声，少女感明显"
    },
    "male": {                   # 男性声音（保留原始或回到男声）
        "pitch_ratio": 1.0,
        "formant_ratio": 1.0,
        "description": "原始声音"
    },
    "female_to_male": {         # 女声变男声
        "pitch_ratio": 0.65,
        "formant_ratio": 0.85,
        "description": "女声变男声"
    }
}

# ============================================================================
# 音频处理配置
# ============================================================================

# 音频采样率
SAMPLE_RATE = None  # None 表示保持原始采样率

# 音频数据类型
AUDIO_DTYPE = "float64"  # WORLD 声码器要求

# ============================================================================
# 文件和路径配置
# ============================================================================

# 输入/输出目录
INPUT_DIR = "input_audio"
OUTPUT_DIR = "output_audio"

# 支持的音频格式
SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# ============================================================================
# 日志配置
# ============================================================================

# 日志级别: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# 是否保存日志到文件
LOG_TO_FILE = True
LOG_FILE = "voice_changer.log"

# ============================================================================
# 处理配置
# ============================================================================

# 是否显示详细进度信息
VERBOSE = True

# 批量处理时的并行工作数（-1 表示自动）
N_JOBS = -1

# ============================================================================
# 质量验证配置
# ============================================================================

# 检查输入文件
CHECK_INPUT = True

# 最小音频时长（秒）- 短于此长度会发出警告
MIN_AUDIO_LENGTH = 3.0

# 最大音频时长（秒）- 长于此长度可能需要分块处理
MAX_AUDIO_LENGTH = 300.0

# 采样率范围
MIN_SAMPLE_RATE = 8000   # 8 kHz
MAX_SAMPLE_RATE = 48000  # 48 kHz

# ============================================================================
# 参数验证范围
# ============================================================================

# pitch_ratio 的有效范围
PITCH_RATIO_MIN = 0.5
PITCH_RATIO_MAX = 2.5

# formant_ratio 的有效范围
FORMANT_RATIO_MIN = 0.8
FORMANT_RATIO_MAX = 1.4

# ============================================================================
# 项目信息
# ============================================================================

PROJECT_NAME = "WORLD 声码器高质量变声系统"
PROJECT_VERSION = "1.0.0"
AUTHOR = "AI 助手"
CREATED_DATE = "2026年5月18日"
DESCRIPTION = "基于 WORLD 声码器的专业级男声变女声系统"

# ============================================================================
# 帮助文本和说明
# ============================================================================

USAGE_GUIDE = """
快速开始指南：

1. 激活虚拟环境：
   voice_env\\Scripts\\activate

2. 选择使用方式：
   a) 交互式菜单（推荐新手）：
      python interactive_demo.py
   
   b) 在 Python 脚本中调用：
      from high_quality_voice_changer import convert_gender_high_quality
      convert_gender_high_quality("input.wav", "output.wav")
   
   c) 运行示例脚本：
      python examples.py

3. 查看详细文档：
   - README.md: 项目概述和参数说明
   - high_quality_voice_changer.py: 核心库文档
   - examples.py: 多个使用示例
"""

# ============================================================================
# 调试配置
# ============================================================================

# 是否启用调试模式
DEBUG = False

# 调试时是否保存中间特征（耗时和占空间，仅用于研究）
SAVE_FEATURES = False
FEATURES_DIR = "debug_features"

# ============================================================================
# 函数：获取配置
# ============================================================================

def get_preset(preset_name):
    """
    获取预设参数
    
    :param preset_name: 预设名称（见 PRESET_PARAMS）
    :return: 参数字典
    """
    if preset_name not in PRESET_PARAMS:
        raise ValueError(f"未知预设：{preset_name}。可用预设：{list(PRESET_PARAMS.keys())}")
    return PRESET_PARAMS[preset_name]


def validate_parameters(pitch_ratio, formant_ratio):
    """
    验证参数的有效范围
    
    :param pitch_ratio: 音调升高倍数
    :param formant_ratio: 共振峰平移倍数
    :return: 如果有效返回 True，否则返回 False 和错误信息
    """
    errors = []
    
    if not PITCH_RATIO_MIN <= pitch_ratio <= PITCH_RATIO_MAX:
        errors.append(f"pitch_ratio 应在 {PITCH_RATIO_MIN} - {PITCH_RATIO_MAX} 之间")
    
    if not FORMANT_RATIO_MIN <= formant_ratio <= FORMANT_RATIO_MAX:
        errors.append(f"formant_ratio 应在 {FORMANT_RATIO_MIN} - {FORMANT_RATIO_MAX} 之间")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "验证通过"


if __name__ == "__main__":
    # 测试配置
    print(f"项目：{PROJECT_NAME}")
    print(f"版本：{PROJECT_VERSION}")
    print(f"\n默认参数：{DEFAULT_PARAMS}")
    print(f"\n可用预设：")
    for name, params in PRESET_PARAMS.items():
        print(f"  {name}: {params['description']}")
    
    # 测试参数验证
    print(f"\n参数验证测试：")
    valid, msg = validate_parameters(1.75, 1.15)
    print(f"  (1.75, 1.15): {valid} - {msg}")
    
    valid, msg = validate_parameters(3.0, 1.5)
    print(f"  (3.0, 1.5): {valid} - {msg}")
