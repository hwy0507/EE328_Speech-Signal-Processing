"""
系统环境检查和测试脚本

这个脚本验证项目环境是否配置正确
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """检查 Python 版本"""
    print("=" * 70)
    print("🐍 Python 版本检查")
    print("=" * 70)
    
    version_info = sys.version_info
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print(f"Python 版本: {version}")
    print(f"Python 执行路径: {sys.executable}")
    
    if version_info.major == 3 and version_info.minor >= 8:
        print("✅ Python 版本满足要求 (>=3.8)")
        return True
    else:
        print("❌ Python 版本过低，建议 >=3.8")
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 70)
    print("📦 依赖包检查")
    print("=" * 70)
    
    required_packages = {
        'pyworld': '0.3.5',
        'librosa': '0.11.0',
        'soundfile': '0.13.1',
        'numpy': '2.4.0'
    }
    
    all_ok = True
    
    for package_name, min_version in required_packages.items():
        try:
            module = __import__(package_name)
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                version = "unknown"
            print(f"✅ {package_name:20s} {version:15s} ✓")
        except ImportError:
            print(f"❌ {package_name:20s} 未安装 ✗")
            all_ok = False
    
    return all_ok


def check_project_files():
    """检查项目文件"""
    print("\n" + "=" * 70)
    print("📁 项目文件检查")
    print("=" * 70)
    
    required_files = [
        "high_quality_voice_changer.py",
        "interactive_demo.py",
        "examples.py",
        "config.py",
        "README.md",
        "QUICK_START.md",
        "requirements.txt"
    ]
    
    all_ok = True
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename:40s} {size:>10,} bytes")
        else:
            print(f"❌ {filename:40s} 未找到")
            all_ok = False
    
    return all_ok


def test_basic_import():
    """测试基本导入"""
    print("\n" + "=" * 70)
    print("🧪 基本导入测试")
    print("=" * 70)
    
    try:
        from high_quality_voice_changer import convert_gender_high_quality, batch_convert
        print("✅ 成功导入 convert_gender_high_quality")
        print("✅ 成功导入 batch_convert")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_audio_file():
    """检查示例音频文件"""
    print("\n" + "=" * 70)
    print("🎵 音频文件检查")
    print("=" * 70)
    
    audio_file = "test_16k.wav"
    
    if os.path.exists(audio_file):
        size = os.path.getsize(audio_file)
        print(f"✅ 找到测试音频文件: {audio_file}")
        print(f"   文件大小: {size:,} bytes ({size/1024:.1f} KB)")
        
        try:
            import librosa
            duration = librosa.get_samplerate(audio_file)  # This will work in later versions
            # 简单验证文件可以被 librosa 加载
            y, sr = librosa.load(audio_file, sr=None)
            print(f"   采样率: {sr} Hz")
            print(f"   时长: {len(y)/sr:.2f} 秒")
            print("✅ 音频文件格式正确")
            return True
        except Exception as e:
            print(f"⚠️  无法解析音频文件: {e}")
            return False
    else:
        print(f"⚠️  未找到测试音频文件 '{audio_file}'")
        print("   你可以将自己的 .wav 文件放在项目目录中进行测试")
        return None


def print_summary(results):
    """打印检查总结"""
    print("\n" + "=" * 70)
    print("📊 检查总结")
    print("=" * 70)
    
    categories = [
        ("Python 版本", results[0]),
        ("依赖包", results[1]),
        ("项目文件", results[2]),
        ("基本导入", results[3]),
        ("音频文件", results[4])
    ]
    
    all_ok = all(r is True for r in results)
    
    for category, result in categories:
        if result is True:
            status = "✅"
        elif result is False:
            status = "❌"
        else:
            status = "⚠️ "
        print(f"{status} {category}")
    
    print("\n" + "=" * 70)
    
    if all_ok:
        print("🎉 所有检查通过！项目已准备好使用")
        print("\n下一步：")
        print("  1. 运行交互式界面: python interactive_demo.py")
        print("  2. 查看使用示例: python examples.py")
        print("  3. 查看文档: 打开 README.md 或 QUICK_START.md")
    else:
        print("⚠️  有些检查未通过，请查看上面的错误信息")
        print("\n建议：")
        print("  - 确保虚拟环境已激活: voice_env\\Scripts\\activate")
        print("  - 重新安装依赖: pip install -r requirements.txt")
        print("  - 检查网络连接")
    
    print("=" * 70)


def main():
    print("\n")
    print("█" * 70)
    print("  🎤 WORLD 声码器变声项目 - 环境检查")
    print("█" * 70)
    print()
    
    results = [
        check_python_version(),
        check_dependencies(),
        check_project_files(),
        test_basic_import(),
        test_audio_file()
    ]
    
    print_summary(results)
    
    # 返回状态码
    if all(r is True for r in results):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
