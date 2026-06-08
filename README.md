# EE328 Speech Signal Processing Repository

Course repository containing both the regular lab homework archive and the final project implementation.

- `Lab-Homework/`: semester lab assignments, reports, and submission bundles
- `final_ui/`: the integrated local GUI application used for final-project demonstration
- `anonymization/`: the standalone voice-anonymization pipeline and evaluation workflow

## Repository Layout

```text
期末proj/
├── README.md
├── Lab-Homework/
├── final_ui/
└── anonymization/
```

## Subprojects

### `Lab-Homework/`

Coursework archive for the regular speech-signal-processing labs.

Typical contents per lab:

- assignment PDF
- submitted report PDF / Markdown / LaTeX source
- MATLAB source (`.m`, `.mlx`)
- packaged submission zip files
- generated figures or audio used in the report

Repository role:

- retained as a course record
- not required by the final anonymization pipeline at runtime
- useful for tracing how earlier lab work connects to the final project

Maintenance note:

- lab folders are treated as mostly frozen archival material
- obvious local build artifacts are ignored by the repository where possible

### `final_ui/`

Integrated demo application for local experimentation and classroom presentation.

Scope:

- GUI-based speech processing workflow
- WORLD-based voice conversion
- FreeVC / OpenVoice integration
- speaker-similarity evaluation hooks
- bridge layer for the anonymization pipeline

Primary entry points:

- `final_ui/代码/voice_gui.py`
- `final_ui/代码/tools/anonymization_runner.py`
- `final_ui/代码/run_gui.bat`
- `final_ui/代码/run_gui.ps1`

Local docs:

- `final_ui/README.md`
- `final_ui/文档/`

### `anonymization/`

Core implementation of the speech-anonymization branch developed in this project.

Scope:

- target voice pool selection
- candidate voice-conversion generation
- `metric_clarity` post-selection strategy
- `PPG-tone` naturalness refinement for Chinese speech
- local privacy/content evaluation and reporting

Primary entry points:

- `anonymization/代码/recording_demo_ui.py`
- `anonymization/代码/run_privacy_optimized_recording.py`
- `anonymization/代码/generate_report_evaluation.py`

Key implementation files:

- `anonymization/代码/privacy_target_optimizer.py`
- `anonymization/代码/ppg_tone_naturalizer.py`
- `anonymization/代码/vc_candidate_builder.py`

Local docs:

- `anonymization/README.md`
- `anonymization/文档/`

## Quick Start

Run the integrated desktop-style UI:

```bash
cd final_ui/代码
python voice_gui.py
```

Run the standalone anonymization web demo:

```bash
cd anonymization/代码
python recording_demo_ui.py --port 7862
```

## Developer Reading Order

For contributors who only need the core implementation path, start from:

1. `final_ui/代码/voice_gui.py`
2. `final_ui/代码/tools/anonymization_runner.py`
3. `anonymization/代码/recording_demo_ui.py`
4. `anonymization/代码/privacy_target_optimizer.py`
5. `anonymization/代码/ppg_tone_naturalizer.py`

## Outputs and Reports

Representative artifacts are kept with the corresponding subproject instead of being lifted to the repository root:

- lab reports and archived submissions: `Lab-Homework/`
- evaluation reports and figures: `anonymization/文档/`
- anonymized demo audio and benchmark material: `anonymization/文档/`
- GUI usage notes and submission-facing docs: `final_ui/文档/`

## Maintenance Notes

- Root-level structure is intentionally minimal; feature-specific assets stay inside their owning subproject.
- `Lab-Homework/` is the archival course-lab layer.
- `final_ui/` is the presentation-facing application layer.
- `anonymization/` is the research/algorithm layer for the privacy pipeline.
- When extending the project, prefer updating subproject-local README and docs first, then keep this root README limited to architecture and navigation.
