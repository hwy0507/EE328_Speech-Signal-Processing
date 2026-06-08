# EE328 Speech Project Repository

这个仓库是课程项目最终整理版，根目录只保留两个子项目：

```text
期末proj/
├── README.md
├── final_ui/
└── anonymization/
```

## 1. `final_ui/`

这是最终展示用的综合图形界面系统，负责把多个语音处理功能整合到一个本地 UI 中。

主要包含：

- WORLD 声码器变声
- FreeVC 声线克隆
- OpenVoice 声线克隆
- Speaker Similarity 评估
- 匿名化处理流程接入

主入口：

- `final_ui/代码/voice_gui.py`
- `final_ui/代码/run_gui.bat`
- `final_ui/代码/run_gui.ps1`

文档说明：

- `final_ui/README.md`
- `final_ui/文档/`

## 2. `anonymization/`

这是音色匿名化子项目的独立核心代码，主要负责：

- 目标音色池选择
- FreeVC 候选生成
- `metric_clarity` 清晰度保护
- `PPG-tone` 中文语调自然化
- 本地隐私/内容评估

主入口：

- `anonymization/代码/recording_demo_ui.py`
- `anonymization/代码/run_privacy_optimized_recording.py`
- `anonymization/代码/generate_report_evaluation.py`

文档说明：

- `anonymization/README.md`
- `anonymization/文档/`

## 3. 推荐阅读顺序

如果只想快速看最核心的课程代码，建议按这个顺序：

1. `final_ui/代码/voice_gui.py`
2. `final_ui/代码/tools/anonymization_runner.py`
3. `anonymization/代码/recording_demo_ui.py`
4. `anonymization/代码/privacy_target_optimizer.py`
5. `anonymization/代码/ppg_tone_naturalizer.py`

## 4. 运行方式

运行最终综合 UI：

```bash
cd final_ui/代码
python voice_gui.py
```

单独运行匿名化网页 demo：

```bash
cd anonymization/代码
python recording_demo_ui.py --port 7862
```

## 5. 说明

这个仓库已经按课程提交场景整理过：

- `final_ui/` 放最终总 UI 和其相关文档
- `anonymization/` 放匿名化子项目和其相关文档

测试脚本、示例脚本和快速文档仍有保留，目的是方便复现和演示；真正的核心主线是上述几个入口文件。
