import pyworld as pw
import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path

# 可选的可视化支持（如果用户未安装 matplotlib，不影响核心功能）
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


def convert_gender_high_quality(input_wav_path, output_wav_path, pitch_ratio=1.8, formant_ratio=1.15):
    """
    使用 WORLD 声码器实现高质量男声变女声
    
    参数说明：
    :param input_wav_path: 输入音频文件路径
    :param output_wav_path: 输出音频文件路径
    :param pitch_ratio: 基频升高倍数 (男变女通常在 1.6 - 2.0 之间)
    :param formant_ratio: 共振峰平移倍数，模拟女性较短的声道 (通常在 1.1 - 1.2 之间)
    
    WORLD 声码器原理：
    - F0 (基频)：决定声音的音高（男声低，女声高）
    - Spectral Envelope (频谱包络)：决定声音的音色和共振峰
    - Aperiodicity (非周期成分)：决定声音里的气声、摩擦音和清音
    
    通过独立修改这三个特征，能实现自然的变声效果
    """
    
    print("=" * 60)
    print("🎙️  高质量 WORLD 声码器变声系统启动")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"输入文件不存在: {input_wav_path}")
    
    print(f"\n📂 输入文件: {input_wav_path}")
    print(f"📂 输出文件: {output_wav_path}")
    print(f"🎚️  音调升高比例: {pitch_ratio}x")
    print(f"🎚️  共振峰平移比例: {formant_ratio}x")
    
    print("\n[步骤 1/4] 正在加载音频并进行高精度浮点转换...")
    # WORLD 声码器要求输入音频数据类型为 float64
    x, fs = librosa.load(input_wav_path, sr=None, dtype=np.float64)
    print(f"✓ 加载完成 | 采样率: {fs} Hz | 时长: {len(x)/fs:.2f}s")
    
    print("\n[步骤 2/4] 正在使用 WORLD 算法解剖语音特征...")
    
    # (a) 提取基频 F0 (使用 DIO 算法，并通过 StoneMask 优化)
    print("  → 提取基频 (F0)...")
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    print(f"    ✓ 基频提取完成 | 帧数: {len(f0)}")
    
    # (b) 提取频谱包络 (使用 CheapTrick 算法)
    print("  → 提取频谱包络 (Spectral Envelope)...")
    sp = pw.cheaptrick(x, f0, t, fs)
    print(f"    ✓ 频谱包络提取完成 | 特征维度: {sp.shape}")
    
    # (c) 提取非周期成分 (使用 D4C 算法)
    print("  → 提取非周期成分 (Aperiodicity)...")
    ap = pw.d4c(x, f0, t, fs)
    print(f"    ✓ 非周期成分提取完成 | 特征维度: {ap.shape}")
    
    print("\n[步骤 3/4] 正在进行变声外科手术 (修改基频和共振峰)...")
    
    # [核心修改 1]：音调变高 —— 直接将基频 F0 乘以一个比例
    print("  → 修改基频 (Pitch Shift)...")
    modified_f0 = f0 * pitch_ratio
    print(f"    ✓ 基频修改完成 | 范围: {np.min(modified_f0[modified_f0>0]):.1f} - {np.max(modified_f0):.1f} Hz")
    
    # [核心修改 2]：音色变女声 —— 缩放频谱包络，模拟女性较短的声道
    # 如果只改 F0 不改包络，听起来就像男生吸了氦气一样假。
    print("  → 修改共振峰 (Formant Shifting)...")
    modified_sp = np.zeros_like(sp)
    num_bins = sp.shape[1]
    
    for i in range(sp.shape[0]):
        # 构造插值函数，将低频的能量推向高频，使得共振峰整体右移
        old_freq_axis = np.arange(num_bins)
        # 通过缩放频率轴来实现包络平移
        new_freq_axis = old_freq_axis / formant_ratio
        
        # 使用线性插值计算新的包络值
        modified_sp[i, :] = np.interp(new_freq_axis, old_freq_axis, sp[i, :])
    
    print(f"    ✓ 共振峰修改完成")
    
    print("\n[步骤 4/4] 正在使用修改后的特征重新合成高质量语音...")
    # 合成变声后的音频
    y = pw.synthesize(modified_f0, modified_sp, ap, fs)
    
    # 创建输出目录如果不存在
    output_dir = os.path.dirname(output_wav_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 导出文件
    sf.write(output_wav_path, y, fs)
    print(f"✓ 合成完成 | 输出文件: {output_wav_path}")
    
    print("\n" + "=" * 60)
    print("🎉 变声完成！已为您转换成自然女声")
    print("=" * 60)
    
    return y, fs


def batch_convert(input_dir, output_dir, pitch_ratio=1.8, formant_ratio=1.15):
    """
    批量处理多个音频文件
    
    :param input_dir: 输入目录（包含 .wav 文件）
    :param output_dir: 输出目录
    :param pitch_ratio: 基频升高倍数
    :param formant_ratio: 共振峰平移倍数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(input_path.glob("*.wav"))
    
    if not wav_files:
        print(f"❌ 在 {input_dir} 中未找到 .wav 文件")
        return
    
    print(f"\n🎵 找到 {len(wav_files)} 个音频文件，开始批量转换...\n")
    
    for i, wav_file in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] 正在处理: {wav_file.name}")
        try:
            output_file = output_path / f"converted_{wav_file.name}"
            convert_gender_high_quality(
                str(wav_file), 
                str(output_file), 
                pitch_ratio=pitch_ratio,
                formant_ratio=formant_ratio
            )
        except Exception as e:
            print(f"❌ 处理失败: {e}")
    
    print(f"\n✅ 批量转换完成！所有文件已保存到: {output_dir}")


def auto_gender_voice_converter(input_wav_path, output_wav_path):
    """
    智能双向变声系统：自动识别男女声，并自适应计算转换参数
    """
    print(f"🎵 正在加载音频: {input_wav_path}")
    x, fs = librosa.load(input_wav_path, sr=None, dtype=np.float64)
    
    print("🔍 正在提取语音特征...")
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    
    # ---------------- 性别判别模块 ----------------
    valid_f0 = f0[f0 > 0]
    
    if len(valid_f0) == 0:
        print("❌ 未检测到有效的人声基频，转换中止。")
        return
        
    # 使用中位数获取说话人的真实基准音高，抗干扰能力更强
    median_f0 = np.median(valid_f0)
    print(f"📊 分析完毕：当前说话人的基频中位数为 {median_f0:.1f} Hz")
    
    # 设定分类阈值为 165 Hz
    if median_f0 < 165:
        detected_gender = "男声 (Male)"
        target_f0 = 220.0  # 目标变为标准女声
        formant_ratio = 1.18 # 声道缩短（共振峰右移）
    else:
        detected_gender = "女声 (Female)"
        target_f0 = 120.0  # 目标变为标准男声
        formant_ratio = 0.85 # 声道拉长（共振峰左移）
        
    # 自适应计算需要改变的比例
    pitch_ratio = target_f0 / median_f0
    print(f"🤖 识别结果：【{detected_gender}】")
    print(f"⚙️ 转换策略：将其转换为对应性别，Pitch Ratio={pitch_ratio:.2f}, Formant Ratio={formant_ratio:.2f}")

    # ---------------- 变声手术模块 ----------------
    modified_f0 = f0 * pitch_ratio
    
    # 修改频谱包络（共振峰平移）
    modified_sp = np.zeros_like(sp)
    num_bins = sp.shape[1]
    
    for i in range(sp.shape[0]):
        old_freq_axis = np.arange(num_bins)
        new_freq_axis = old_freq_axis / formant_ratio
        # 线性插值
        modified_sp[i, :] = np.interp(new_freq_axis, old_freq_axis, sp[i, :])
    
    print("✨ 正在合成新语音...")
    y = pw.synthesize(modified_f0, modified_sp, ap, fs)
    
    sf.write(output_wav_path, y, fs)
    print(f"✅ 转换完成！输出文件: {output_wav_path}\n")
    
    return y, fs


def auto_gender_voice_converter_with_plot(input_wav_path, output_wav_path, show_seconds=2):
    """
    auto_gender_voice_converter 的可视化版本，会绘制基频直方图并标注阈值和中位数。
    如果未安装 matplotlib，则行为等同于 auto_gender_voice_converter。
    """
    if not _HAVE_MPL:
        print("⚠️ matplotlib 未安装，跳过可视化。")
        return auto_gender_voice_converter(input_wav_path, output_wav_path)

    print(f"🎵 正在加载音频: {input_wav_path}")
    x, fs = librosa.load(input_wav_path, sr=None, dtype=np.float64)
    
    print("🔍 正在提取语音特征...")
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    
    valid_f0 = f0[f0 > 0]
    if len(valid_f0) == 0:
        print("❌ 未检测到有效的人声基频，转换中止。")
        return

    median_f0 = np.median(valid_f0)
    threshold = 165.0
    if median_f0 < threshold:
        detected_gender = "男声"
        target_f0 = 220.0
        formant_ratio = 1.18
    else:
        detected_gender = "女声"
        target_f0 = 120.0
        formant_ratio = 0.85

    pitch_ratio = target_f0 / median_f0

    print("-" * 30)
    print(f"📊 基频中位数: {median_f0:.1f} Hz")
    print(f"🤖 识别结果: 【{detected_gender}】")
    print(f"⚙️ 目标策略: 转换为异性 (Pitch Ratio: {pitch_ratio:.2f})")
    print("-" * 30)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.hist(valid_f0, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='green', linestyle='solid', linewidth=2, label=f'男女声分界阈值: {threshold} Hz')
    plt.axvline(median_f0, color='red', linestyle='dashed', linewidth=2, label=f'当前音频基频中位数: {median_f0:.1f} Hz\n判定为: {detected_gender}')
    plt.title('语音基频 (F_0) 分布与性别识别结果', fontsize=16)
    plt.xlabel('频率 (Hz)', fontsize=12)
    plt.ylabel('出现频次 (帧数)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.show(block=False)
    plt.pause(show_seconds)

    # 执行变声
    print("\n✨ 正在合成新语音，请稍候...")
    modified_f0 = f0 * pitch_ratio
    modified_sp = np.zeros_like(sp)
    num_bins = sp.shape[1]
    for i in range(sp.shape[0]):
        old_freq_axis = np.arange(num_bins)
        new_freq_axis = old_freq_axis / formant_ratio
        modified_sp[i, :] = np.interp(new_freq_axis, old_freq_axis, sp[i, :])

    y = pw.synthesize(modified_f0, modified_sp, ap, fs)
    sf.write(output_wav_path, y, fs)
    print(f"✅ 转换完成！输出文件: {output_wav_path}")

    plt.show()
    return y, fs


if __name__ == "__main__":
    # 示例用法
    print("\n🎤 WORLD 声码器高质量变声系统")
    print("=" * 60)
    print("参数调节建议：")
    print("  - pitch_ratio: 1.6-2.0 (默认 1.8) | 更高=更尖锐的女声")
    print("  - formant_ratio: 1.1-1.2 (默认 1.15) | 更高=音色更接近女性")
    print("=" * 60)
    
    # 单个文件转换示例
    # input_file = "your_male_voice.wav"
    # output_file = "converted_female_voice.wav"
    # convert_gender_high_quality(input_file, output_file, pitch_ratio=1.75, formant_ratio=1.18)
    
    # 或者批量转换
    # batch_convert("input_audio", "output_audio", pitch_ratio=1.75, formant_ratio=1.18)
    
    print("\n💡 使用方法：")
    print("  1. 单个文件: convert_gender_high_quality('input.wav', 'output.wav')")
    print("  2. 批量处理: batch_convert('input_dir', 'output_dir')")
    print("\n请在脚本中修改文件路径后运行")
