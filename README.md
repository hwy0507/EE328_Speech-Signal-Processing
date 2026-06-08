# EE328 Speech Signal Processing Repository

Course repository containing both the regular lab homework archive and the final project implementation.

- `Lab-Homework/`: semester lab assignments, reports, and submission bundles
- `final_project/`: final-project source tree

## Repository Layout

```text
期末proj/
├── README.md
├── Lab-Homework/
└── final_project/
    ├── final_ui/
    └── anonymization/
```

## Top-Level Modules

### `Lab-Homework/`

Coursework archive for the regular speech-signal-processing labs.

Typical contents per lab:

- assignment PDF
- submitted report PDF / Markdown / LaTeX source
- MATLAB source (`.m`, `.mlx`)
- packaged submission zip files
- generated figures or audio used in the report

Current coverage:

- `lab1` to `lab10`

Repository role:

- retained as a course record
- not required by the final anonymization pipeline at runtime
- useful for tracing how earlier lab work connects to the final project

Maintenance note:

- lab folders are treated as mostly frozen archival material
- obvious local build artifacts are ignored by the repository where possible

### `final_project/`

Container directory for the final course project. It groups the presentation-facing application and the anonymization research pipeline under a single project root.

Submodules:

- `final_project/final_ui/`
- `final_project/anonymization/`

## Final Project Submodules

### `final_project/final_ui/`

Integrated demo application for local experimentation and classroom presentation.

Scope:

- GUI-based speech processing workflow
- WORLD-based voice conversion
- FreeVC / OpenVoice integration
- speaker-similarity evaluation hooks
- bridge layer for the anonymization pipeline

Primary entry points:

- `final_project/final_ui/代码/voice_gui.py`
- `final_project/final_ui/代码/tools/anonymization_runner.py`
- `final_project/final_ui/代码/run_gui.bat`
- `final_project/final_ui/代码/run_gui.ps1`

Local docs:

- `final_project/final_ui/README.md`
- `final_project/final_ui/文档/`

### `final_project/anonymization/`

Core implementation of the speech-anonymization branch developed in this project.

Scope:

- target voice pool selection
- candidate voice-conversion generation
- `metric_clarity` post-selection strategy
- `PPG-tone` naturalness refinement for Chinese speech
- local privacy/content evaluation and reporting

Primary entry points:

- `final_project/anonymization/代码/recording_demo_ui.py`
- `final_project/anonymization/代码/run_privacy_optimized_recording.py`
- `final_project/anonymization/代码/generate_report_evaluation.py`

Key implementation files:

- `final_project/anonymization/代码/privacy_target_optimizer.py`
- `final_project/anonymization/代码/ppg_tone_naturalizer.py`
- `final_project/anonymization/代码/vc_candidate_builder.py`

Local docs:

- `final_project/anonymization/README.md`
- `final_project/anonymization/文档/`

## Quick Start

Run the integrated desktop-style UI:

```bash
cd final_project/final_ui/代码
python voice_gui.py
```

Run the standalone anonymization web demo:

```bash
cd final_project/anonymization/代码
python recording_demo_ui.py --port 7862
```

## Developer Reading Order

For contributors who only need the core implementation path, start from:

1. `final_project/final_ui/代码/voice_gui.py`
2. `final_project/final_ui/代码/tools/anonymization_runner.py`
3. `final_project/anonymization/代码/recording_demo_ui.py`
4. `final_project/anonymization/代码/privacy_target_optimizer.py`
5. `final_project/anonymization/代码/ppg_tone_naturalizer.py`

## Outputs and Reports

Representative artifacts are kept with the corresponding module instead of being lifted to the repository root:

- lab reports and archived submissions: `Lab-Homework/`
- evaluation reports and figures: `final_project/anonymization/文档/`
- anonymized demo audio and benchmark material: `final_project/anonymization/文档/`
- GUI usage notes and submission-facing docs: `final_project/final_ui/文档/`

## Maintenance Notes

- Root-level structure is intentionally minimal and split by course function.
- `Lab-Homework/` is the archival course-lab layer.
- `final_project/` is the final submission layer.
- `final_project/final_ui/` is the presentation-facing application layer.
- `final_project/anonymization/` is the research/algorithm layer for the privacy pipeline.
- When extending the project, prefer updating submodule-local README and docs first, then keep this root README limited to architecture and navigation.
