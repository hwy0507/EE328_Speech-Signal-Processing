# final_ui

这个目录现在按用途整理成两部分：

```text
final_ui/
├── 代码/   # 最终 UI 主程序、依赖脚本、外部匿名化融合代码
├── 文档/   # 说明文档、交接材料、结果样例
├── README.md
├── .gitignore
└── .gitattributes
```

## 代码

主入口在：

- `代码/voice_gui.py`
- `代码/run_gui.bat`
- `代码/run_gui.ps1`

核心依赖包括：

- `代码/tools/`：FreeVC、OpenVoice、speaker similarity、匿名化融合 runner
- `代码/_external/EE328_Speech-Signal-Processing/`：匿名化外部项目依赖
- `代码/playback_worker.py`、`代码/record_chinese.py`：播放与录音
- `代码/config.py`、`代码/requirements*.txt`：配置与环境依赖

如果只启动最终 GUI，建议从 `代码/` 目录运行：

```bash
cd 代码
python voice_gui.py
```

Windows 下也可以直接运行：

```text
代码/run_gui.bat
代码/run_gui.ps1
```

## 文档

说明和交接材料在 `文档/`：

- `文档/README_旧版.md`：整理前的完整项目说明
- `文档/QUICK_START.md`：快速上手
- `文档/HANDOFF.md`：交接说明
- `文档/SUBMISSION_NOTE.md`：提交包说明
- `文档/部署完成总结.md`：部署总结
- `文档/结果样例/`：相似度评估 JSON 样例

## 备注

当前目录中没有直接提供可用的 `env_map.json`。如果要使用 FreeVC、OpenVoice 或匿名化融合功能，需要在 `代码/` 下按旧版文档说明自行准备本机路径配置。
