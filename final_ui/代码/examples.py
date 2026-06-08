"""
WORLD 声码器变声项目 - 使用示例脚本

这个脚本展示了如何在实际项目中使用 high_quality_voice_changer 库
"""

from high_quality_voice_changer import convert_gender_high_quality, batch_convert
import os


def example_1_basic_conversion():
    """
    示例 1：最简单的单文件转换
    使用默认参数，一行代码即可完成变声
    """
    print("\n" + "="*70)
    print("示例 1：基础单文件转换")
    print("="*70)
    
    # 假设你有一个输入文件 "my_voice.wav"
    input_file = "my_voice.wav"
    output_file = "my_voice_converted.wav"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 示例文件 '{input_file}' 不存在")
        print("💡 提示：请将你的音频文件放在项目目录下，然后修改文件名重试")
        return
    
    try:
        # 使用默认参数进行转换
        convert_gender_high_quality(input_file, output_file)
        print(f"✅ 转换完成！输出文件：{output_file}")
    except Exception as e:
        print(f"❌ 转换失败：{e}")


def example_2_custom_parameters():
    """
    示例 2：使用自定义参数
    根据你的喜好调整音调和音色
    """
    print("\n" + "="*70)
    print("示例 2：自定义参数转换")
    print("="*70)
    
    input_file = "my_voice.wav"
    
    if not os.path.exists(input_file):
        print(f"❌ 示例文件 '{input_file}' 不存在")
        return
    
    # 不同的参数组合，产生不同的效果
    configs = [
        {
            "name": "轻微女化",
            "output": "converted_light_female.wav",
            "pitch_ratio": 1.65,
            "formant_ratio": 1.12
        },
        {
            "name": "自然女声（推荐）",
            "output": "converted_natural_female.wav",
            "pitch_ratio": 1.75,
            "formant_ratio": 1.15
        },
        {
            "name": "高亢少女",
            "output": "converted_young_female.wav",
            "pitch_ratio": 1.90,
            "formant_ratio": 1.18
        }
    ]
    
    for config in configs:
        print(f"\n正在转换：{config['name']}")
        print(f"  参数：pitch_ratio={config['pitch_ratio']}, formant_ratio={config['formant_ratio']}")
        
        try:
            convert_gender_high_quality(
                input_file,
                config['output'],
                pitch_ratio=config['pitch_ratio'],
                formant_ratio=config['formant_ratio']
            )
            print(f"✅ 完成！输出：{config['output']}")
        except Exception as e:
            print(f"❌ 失败：{e}")


def example_3_batch_processing():
    """
    示例 3：批量处理目录中的所有音频
    适合处理多个文件时使用
    """
    print("\n" + "="*70)
    print("示例 3：批量处理")
    print("="*70)
    
    input_dir = "input_audios"  # 输入目录
    output_dir = "output_audios"  # 输出目录
    
    # 创建输入目录（如果不存在）
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录 '{input_dir}' 不存在")
        print(f"💡 提示：创建目录 '{input_dir}'，然后放入你的 .wav 文件")
        return
    
    try:
        batch_convert(
            input_dir,
            output_dir,
            pitch_ratio=1.75,
            formant_ratio=1.15
        )
        print(f"✅ 批量处理完成！输出目录：{output_dir}")
    except Exception as e:
        print(f"❌ 处理失败：{e}")


def example_4_female_to_male():
    """
    示例 4：女声变男声
    通过反向参数配置实现女→男转换
    """
    print("\n" + "="*70)
    print("示例 4：女声变男声")
    print("="*70)
    
    input_file = "female_voice.wav"
    output_file = "female_to_male.wav"
    
    if not os.path.exists(input_file):
        print(f"❌ 示例文件 '{input_file}' 不存在")
        print("💡 提示：请提供一个女性声音的音频文件来测试女→男转换")
        return
    
    try:
        # 女→男：降低基频和共振峰
        convert_gender_high_quality(
            input_file,
            output_file,
            pitch_ratio=0.65,      # 降低基频（女声更高，所以要降低）
            formant_ratio=0.85     # 降低共振峰（女性声道短，男性长）
        )
        print(f"✅ 女声变男声完成！输出：{output_file}")
    except Exception as e:
        print(f"❌ 转换失败：{e}")


def example_5_advanced_comparison():
    """
    示例 5：高级用法 - 对比不同参数效果
    通过生成多个版本来找到最佳参数
    """
    print("\n" + "="*70)
    print("示例 5：参数对比实验")
    print("="*70)
    
    input_file = "my_voice.wav"
    
    if not os.path.exists(input_file):
        print(f"❌ 示例文件 '{input_file}' 不存在")
        return
    
    # 创建一个参数矩阵，测试不同的组合
    pitch_ratios = [1.65, 1.75, 1.85]
    formant_ratios = [1.12, 1.15, 1.18]
    
    print(f"将生成 {len(pitch_ratios) * len(formant_ratios)} 个变声版本...")
    print("参数范围：")
    print(f"  pitch_ratio: {pitch_ratios}")
    print(f"  formant_ratio: {formant_ratios}")
    
    results = []
    
    for pitch in pitch_ratios:
        for formant in formant_ratios:
            output_file = f"test_p{pitch:.2f}_f{formant:.2f}.wav"
            
            try:
                print(f"\n处理：pitch={pitch}, formant={formant}")
                convert_gender_high_quality(
                    input_file,
                    output_file,
                    pitch_ratio=pitch,
                    formant_ratio=formant
                )
                results.append({
                    "pitch": pitch,
                    "formant": formant,
                    "output": output_file,
                    "status": "✅"
                })
            except Exception as e:
                print(f"❌ 失败：{e}")
                results.append({
                    "pitch": pitch,
                    "formant": formant,
                    "output": output_file,
                    "status": "❌"
                })
    
    # 汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    for result in results:
        print(f"{result['status']} pitch={result['pitch']:.2f}, "
              f"formant={result['formant']:.2f} → {result['output']}")
    
    print("\n💡 提示：比较这些文件的听感，找到最满意的参数组合！")


def main():
    """主程序"""
    print("\n")
    print("█" * 70)
    print("  🎤 WORLD 声码器变声项目 - 使用示例")
    print("█" * 70)
    
    print("\n选择要运行的示例：")
    print("  [1] 示例 1：基础单文件转换")
    print("  [2] 示例 2：自定义参数转换")
    print("  [3] 示例 3：批量处理")
    print("  [4] 示例 4：女声变男声")
    print("  [5] 示例 5：参数对比实验")
    print("  [0] 退出")
    
    while True:
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == "1":
            example_1_basic_conversion()
        elif choice == "2":
            example_2_custom_parameters()
        elif choice == "3":
            example_3_batch_processing()
        elif choice == "4":
            example_4_female_to_male()
        elif choice == "5":
            example_5_advanced_comparison()
        elif choice == "0":
            print("\n👋 感谢使用！再见！\n")
            break
        else:
            print("❌ 无效选择，请重试")


if __name__ == "__main__":
    # 如果这个脚本被直接运行，显示菜单
    main()
    
    # 如果要直接运行某个示例，可以取消注释下面的代码：
    # example_1_basic_conversion()
    # example_2_custom_parameters()
    # example_3_batch_processing()
