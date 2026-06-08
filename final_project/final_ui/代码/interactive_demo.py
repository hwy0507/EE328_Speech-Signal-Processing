"""
WORLD 声码器高质量变声项目 - 交互式测试脚本

这个脚本提供了一个友好的界面来使用高质量变声功能
"""

import os
import sys
from pathlib import Path
from high_quality_voice_changer import convert_gender_high_quality, batch_convert


def main_menu():
    """主菜单"""
    while True:
        print("\n" + "=" * 70)
        print("🎤 WORLD 声码器高质量变声系统 - 交互式菜单")
        print("=" * 70)
        print("\n选择操作：")
        print("  [1] 转换单个音频文件")
        print("  [2] 批量转换目录中的所有音频文件")
        print("  [3] 自定义参数并转换")
        print("  [4] 显示参数调节指南")
        print("  [5] 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            convert_single_file()
        elif choice == "2":
            batch_convert_files()
        elif choice == "3":
            custom_convert()
        elif choice == "4":
            show_parameter_guide()
        elif choice == "5":
            print("\n👋 感谢使用！再见！")
            sys.exit(0)
        else:
            print("❌ 无效选择，请重试")


def convert_single_file():
    """转换单个文件"""
    print("\n" + "-" * 70)
    print("📁 单个文件转换")
    print("-" * 70)
    
    input_file = input("\n请输入音频文件路径 (如: voice.wav): ").strip()
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    # 生成输出文件名
    base_name = Path(input_file).stem
    output_file = f"{base_name}_converted.wav"
    
    output_path = input(f"\n输出文件路径 (默认: {output_file}): ").strip()
    if output_path:
        output_file = output_path
    
    print(f"\n使用默认参数: pitch_ratio=1.8, formant_ratio=1.15")
    try:
        convert_gender_high_quality(input_file, output_file)
        print(f"\n✅ 成功！文件已保存: {output_file}")
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")


def batch_convert_files():
    """批量转换文件"""
    print("\n" + "-" * 70)
    print("📂 批量转换")
    print("-" * 70)
    
    input_dir = input("\n请输入输入目录 (包含 .wav 文件): ").strip()
    
    if not os.path.isdir(input_dir):
        print(f"❌ 目录不存在: {input_dir}")
        return
    
    output_dir = input("\n请输入输出目录: ").strip()
    
    print(f"\n使用默认参数: pitch_ratio=1.8, formant_ratio=1.15")
    try:
        batch_convert(input_dir, output_dir)
        print(f"\n✅ 批量转换完成！")
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")


def custom_convert():
    """自定义参数转换"""
    print("\n" + "-" * 70)
    print("⚙️  自定义参数转换")
    print("-" * 70)
    
    input_file = input("\n请输入音频文件路径: ").strip()
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    try:
        pitch_ratio = float(input("\n音调升高倍数 (推荐 1.6-2.0, 默认 1.8): ") or 1.8)
        formant_ratio = float(input("共振峰平移倍数 (推荐 1.1-1.2, 默认 1.15): ") or 1.15)
    except ValueError:
        print("❌ 参数格式错误")
        return
    
    base_name = Path(input_file).stem
    output_file = f"{base_name}_converted.wav"
    output_path = input(f"\n输出文件路径 (默认: {output_file}): ").strip()
    if output_path:
        output_file = output_path
    
    try:
        convert_gender_high_quality(input_file, output_file, pitch_ratio, formant_ratio)
        print(f"\n✅ 成功！文件已保存: {output_file}")
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")


def show_parameter_guide():
    """显示参数调节指南"""
    print("\n" + "=" * 70)
    print("📚 WORLD 声码器参数调节指南")
    print("=" * 70)
    
    guide = """
📊 参数说明：

1️⃣  pitch_ratio (音调升高倍数)
   ├─ 范围: 1.5 - 2.2
   ├─ 推荐: 1.6 - 2.0 (男变女)
   ├─ 1.6-1.7  : 轻微女化，保留部分男性特征
   ├─ 1.75-1.85: 自然女声，最平衡的效果 ⭐
   ├─ 1.9-2.0  : 高亢女声，少女感明显
   └─ >2.0     : 极高音，可能显得不自然

2️⃣  formant_ratio (共振峰平移倍数)
   ├─ 范围: 1.05 - 1.25
   ├─ 推荐: 1.1 - 1.2 (男变女)
   ├─ 1.10-1.12: 保守平移，保留部分男性音色
   ├─ 1.13-1.18: 自然女性音色，最接近真实 ⭐
   ├─ 1.19-1.22: 明显女性特征，但可能过度
   └─ >1.23    : 可能会导致失真或不自然

🎯 参数组合建议：

  ┌─────────────────────────────────────────┐
  │ 场景：轻微女化（保留特色）              │
  │ pitch_ratio = 1.65, formant_ratio = 1.12 │
  └─────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │ 场景：自然女声（推荐首选）⭐            │
  │ pitch_ratio = 1.75, formant_ratio = 1.15 │
  └─────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │ 场景：高亢少女感（戏剧化）              │
  │ pitch_ratio = 1.90, formant_ratio = 1.18 │
  └─────────────────────────────────────────┘

💡 调参技巧：

  1. 先用默认参数 (1.8, 1.15) 测试效果
  2. 觉得太尖锐？降低 pitch_ratio (如 1.70)
  3. 觉得音色还是很男性？增加 formant_ratio (如 1.18)
  4. 多做几次测试，找到最适合你的"甜区" (Sweet Spot)
  5. 同一个参数对不同的输入音频效果可能不同

⚠️  质量因素：

  ✓ 输入音频质量高 (清晰、低噪音) → 输出质量越好
  ✓ 输入音频时长 (>3秒) → 基频提取更准确
  ✗ 背景噪音、回声 → 会影响转换效果
  ✗ 极端音调波动 → 可能导致提取失败
"""
    
    print(guide)
    input("\n按 Enter 返回菜单...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 已取消操作，再见！")
        sys.exit(0)
