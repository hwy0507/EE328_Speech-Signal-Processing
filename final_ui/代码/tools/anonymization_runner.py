from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _force_utf8_stdio() -> None:
    """Avoid Windows GBK console/pipe crashes when printing emoji or Chinese text."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1] / "_external" / "EE328_Speech-Signal-Processing"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env_map() -> dict[str, Any]:
    map_path = _repo_root() / "env_map.json"
    if not map_path.exists():
        return {}
    try:
        return json.loads(map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_vc_python() -> Path:
    """Resolve the Python executable used by the real FreeVC/TTS backend.

    The GUI itself may run in a Python 3.12 environment for PyQt/OpenVoice glue code,
    while Coqui TTS 0.22 / FreeVC is installed in the dedicated Python 3.10 env.
    If we do not forward that Python through VC_PYTHON, the external project falls
    back to its DSP placeholder and the result sounds almost unchanged.
    """
    candidates = [
        os.environ.get("VC_PYTHON"),
        _load_env_map().get("语音匿名化（FreeVC）"),
        _load_env_map().get("声线克隆（双音频）"),
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def _resolve_pipeline_python() -> Path:
    """Resolve Python used to execute the teammate anonymization pipeline itself.

    Both the legacy `run_pipeline.py` and the newer `recording_demo_ui.py` import
    audio/ML dependencies that are installed in the dedicated TTS/FreeVC Python
    3.10 environment on this machine.  Running those scripts with the GUI Python
    3.12 environment makes the new path fail before it can produce results.
    """
    candidates = [
        os.environ.get("VOICEPRIVACY_PYTHON"),
        os.environ.get("PIPELINE_PYTHON"),
        str(_resolve_vc_python()),
        sys.executable,
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip() or "audio"
    return "".join(ch if ch.isalnum() or ch in "._-+^()[]{}中文測試標準錄音绿色" else "_" for ch in stem)


def _external_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    vc_python = _resolve_vc_python()
    env["VC_PYTHON"] = str(vc_python)
    env.setdefault("ASV_PYTHON", str(vc_python))
    env.setdefault("VOICEPRIVACY_PYTHON", str(vc_python))
    env.setdefault("PIPELINE_PYTHON", str(vc_python))
    print(f"ℹ️ FreeVC/TTS 后端 Python: {vc_python}", flush=True)

    local_ffmpeg_bin = _repo_root() / "ffmpeg" / "bin"
    if local_ffmpeg_bin.exists():
        env["PATH"] = str(local_ffmpeg_bin) + os.pathsep + env.get("PATH", "")
        env.setdefault("FFMPEG_BINARY", str(local_ffmpeg_bin / "ffmpeg.exe"))
        env.setdefault("FFPROBE_BINARY", str(local_ffmpeg_bin / "ffprobe.exe"))
        print(f"ℹ️ 已注入本项目 ffmpeg 路径: {local_ffmpeg_bin}", flush=True)
    return env


def _stream_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line, flush=True)

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    if stderr:
        for line in stderr.splitlines():
            if line.strip():
                print(line, file=sys.stderr, flush=True)
    if rc != 0:
        detail = "\n".join(line for line in stderr.splitlines() if line.strip())
        if len(detail) > 4000:
            detail = detail[-4000:]
        if detail:
            raise RuntimeError(f"外部匿名化流程失败，returncode={rc}；stderr: {detail}")
        raise RuntimeError(f"外部匿名化流程失败，returncode={rc}")


def _legacy_vc_target_configs(project_root: Path) -> list[Path]:
    """Return all available legacy VC target-pool configs in stable UI order.

    The original upstream `run_pipeline.py` defaults to only
    `vc_target_pool_male.json`, which makes the GUI expose just one anonymous
    voice.  Passing every available config keeps the external project unchanged
    while enabling male/female/general variants when those JSON files exist.
    """
    preferred_names = [
        "vc_target_pool_male.json",
        "vc_target_pool_female.json",
        "vc_target_pool.json",
    ]
    configs: list[Path] = []
    seen: set[Path] = set()
    for name in preferred_names:
        path = (project_root / name).resolve()
        if path.exists() and path not in seen:
            configs.append(path)
            seen.add(path)
    for path in sorted(project_root.glob("vc_target_pool*.json")):
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            configs.append(resolved)
            seen.add(resolved)
    return configs


def _legacy_supported_denoise_presets(script: Path) -> set[str] | None:
    """Best-effort static detection of denoise choices accepted by legacy CLI.

    Returning None means the legacy entry appears unrestricted or could not be
    inspected, so the caller should preserve the user's requested preset.
    """
    try:
        text = script.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    marker = re.search(r"--denoise-preset[\s\S]{0,700}?choices\s*=\s*\[([^\]]+)\]", text)
    if not marker:
        if "DENOISE_PRESETS" not in text:
            return None
        presets_file = script.parent / "audio_preprocess.py"
        try:
            tree = ast.parse(presets_file.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "DENOISE_PRESETS" for target in node.targets):
                continue
            if not isinstance(node.value, ast.Dict):
                return None
            keys = {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            return keys or None
        return None
    choices = set(re.findall(r"['\"]([^'\"]+)['\"]", marker.group(1)))
    return choices or None


def _run_pipeline(project_root: Path, work_root: Path, source: Path, denoise_preset: str) -> None:
    script = project_root / "run_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"未找到匿名化处理入口: {script}")

    legacy_denoise_preset = denoise_preset
    supported_denoise_presets = _legacy_supported_denoise_presets(script)
    if supported_denoise_presets is not None and legacy_denoise_preset not in supported_denoise_presets:
        fallback_preset = "standard" if "standard" in supported_denoise_presets else sorted(supported_denoise_presets)[0]
        print(
            f"ℹ️ 旧版 pipeline 不支持降噪档 {legacy_denoise_preset!r}，"
            f"已自动改用 {fallback_preset!r}。支持项: {sorted(supported_denoise_presets)}",
            flush=True,
        )
        legacy_denoise_preset = fallback_preset

    vc_config_args: list[str] = []
    vc_configs = _legacy_vc_target_configs(project_root)
    for config in vc_configs:
        vc_config_args.extend(["--vc-target-config", str(config)])
    if vc_configs:
        print(
            "ℹ️ 旧版 pipeline 将生成以下匿名化声线:",
            ", ".join(path.name for path in vc_configs),
            flush=True,
        )

    pipeline_python = _resolve_pipeline_python()
    cmd = [
        str(pipeline_python),
        str(script),
        "--project-root",
        str(project_root),
        "--work-root",
        str(work_root),
        "--denoise-preset",
        legacy_denoise_preset,
        *vc_config_args,
        str(source),
    ]
    print(f"ℹ️ 匿名化 pipeline Python: {pipeline_python}", flush=True)
    print("📡 启动匿名化 pipeline:", " ".join(cmd), flush=True)
    _stream_subprocess(cmd, project_root, _external_env())


def _resolve_metric_python() -> Path:
    """Resolve Python for teammate VoicePrivacy-style metrics.

    The upstream metric script needs `speechbrain` for ASV embeddings and
    `faster_whisper` for ASR WER.  In this fused project those packages are most
    likely installed in the same Python 3.10 environment used by FreeVC/TTS, so
    use that environment unless the user explicitly provides ASV_PYTHON.
    """
    candidates = [
        os.environ.get("ASV_PYTHON"),
        os.environ.get("VOICEPRIVACY_PYTHON"),
        _load_env_map().get("语音匿名化（FreeVC）"),
        _load_env_map().get("声线克隆（双音频）"),
        str(_resolve_vc_python()),
        sys.executable,
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def _selection_glob_for_project(project_root: Path, work_root: Path) -> str:
    selection_pattern = work_root / "final" / "preferred_variants" / "*_selections.json"
    try:
        rel = os.path.relpath(selection_pattern, project_root)
        if not rel.startswith("..") and not os.path.isabs(rel):
            return Path(rel).as_posix()
    except Exception:
        pass
    return selection_pattern.as_posix()


def _copy_metric_source(project_root: Path, source: Path, stem: str | None = None) -> Path:
    mirror_name = f"{stem}{source.suffix}" if stem else source.name
    mirror = project_root / mirror_name
    try:
        if mirror.resolve() == source.resolve():
            return mirror
    except Exception:
        pass
    if not mirror.exists() or mirror.stat().st_size != source.stat().st_size:
        shutil.copy2(source, mirror)
        print(f"ℹ️ 已为同学指标脚本复制源音频到项目根目录: {mirror}", flush=True)
    return mirror


def _ensure_metric_source_visible(project_root: Path, source: Path, selection_dir: Path | None = None) -> None:
    """Make source resolvable by the unchanged teammate metric script.

    `evaluate_voiceprivacy.py` resolves source audio by searching files directly
    under `project_root` with the same stem as selection `source_name`.  The GUI
    can pass an arbitrary absolute source outside the cloned teammate project, so
    we mirror the source file into the teammate project root before invoking the
    script.  This keeps the upstream script itself unchanged.
    """
    _copy_metric_source(project_root, source)

    # Older upstream `run_pipeline.py` writes `source_name` after preprocessing,
    # e.g. "標準錄音_5_denoised".  The metric script strips only the
    # "_denoised" suffix and then searches `project_root` for a file with that
    # exact stem, so simply copying the original "標準錄音 5.wav" is not enough
    # when spaces were normalized to underscores.  Mirror the same original
    # audio under every expected stem found in the selection JSON files.
    if selection_dir is None or not selection_dir.exists():
        return
    expected_stems: set[str] = set()
    for selection_path in selection_dir.glob("*_selections.json"):
        try:
            items = json.loads(selection_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source_name = str(item.get("source_name") or "").strip()
            if not source_name:
                continue
            expected_stems.add(source_name.removesuffix("_denoised"))
    for stem in sorted(expected_stems):
        if stem and stem != source.stem:
            _copy_metric_source(project_root, source, stem=stem)


def _augment_report_with_voiceprivacy_metrics(project_root: Path, work_root: Path, source: Path, report: dict[str, Any], report_path: Path) -> None:
    """Run teammate `evaluate_voiceprivacy.py` and merge its metrics into report.

    `run_pipeline.py` only writes acoustic selection metrics (`score`,
    `spectral_distance_db`, `envelope_corr`).  The ASV-EER / ASR-WER style
    metrics that the GUI table expects live in the separate upstream script
    `evaluate_voiceprivacy.py`; without this step Source sim / Sim drop / ASR WER
    are necessarily empty.  We keep that script unmodified and merge its JSON
    output back into `pipeline_report.json` so downstream GUI code can read it.
    """
    script = project_root / "evaluate_voiceprivacy.py"
    selection_dir = work_root / "final" / "preferred_variants"
    if not script.exists() or not selection_dir.exists() or not list(selection_dir.glob("*_selections.json")):
        return

    output_path = work_root / "voiceprivacy_style_results.json"
    metric_python = _resolve_metric_python()
    _ensure_metric_source_visible(project_root, source, selection_dir)

    cmd = [
        str(metric_python),
        str(script),
        "--project-root",
        str(project_root),
        "--selection-glob",
        _selection_glob_for_project(project_root, work_root),
        "--output-path",
        str(output_path),
        "--asv-python",
        str(metric_python),
    ]
    speaker_model = os.environ.get("VOICEPRIVACY_SPEAKER_MODEL") or os.environ.get("SPEAKER_MODEL")
    whisper_model = os.environ.get("VOICEPRIVACY_WHISPER_MODEL") or os.environ.get("WHISPER_MODEL")
    if speaker_model:
        cmd.extend(["--speaker-model", str(Path(speaker_model).expanduser())])
    if whisper_model:
        cmd.extend(["--whisper-model", str(Path(whisper_model).expanduser())])

    print("📊 启动同学仓库 VoicePrivacy-style 指标计算:", " ".join(cmd), flush=True)
    try:
        _stream_subprocess(cmd, project_root, _external_env())
        metrics = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        message = (
            "同学仓库 evaluate_voiceprivacy.py 指标计算失败；匿名音频已生成，但 ASV-EER/ASR-WER "
            f"等指标无法补全。请检查 speechbrain/faster_whisper/本地模型环境。错误: {exc}"
        )
        print(f"⚠️ {message}", file=sys.stderr, flush=True)
        report["voiceprivacy_style_error"] = message
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    report["voiceprivacy_style_results"] = metrics
    report["voiceprivacy_style_report"] = str(output_path)
    if isinstance(metrics, dict):
        if isinstance(metrics.get("baseline"), dict):
            report["baseline"] = metrics["baseline"]
        variant_metrics = metrics.get("variants")
        if isinstance(variant_metrics, dict):
            report["variants"] = variant_metrics
            for variant_name, variant_metric in variant_metrics.items():
                variant_data = (report.get("vc_variants") or {}).get(variant_name)
                if isinstance(variant_data, dict) and isinstance(variant_metric, dict):
                    variant_data.update(variant_metric)

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ VoicePrivacy-style 指标已合并到 pipeline_report: {report_path}", flush=True)


def _new_entry_script(project_root: Path) -> Path | None:
    """Return the latest privacy-optimized entry if the teammate project has it."""
    for name in ("run_privacy_optimized_recording.py", "recording_demo_ui.py"):
        script = project_root / name
        if not script.exists():
            continue
        try:
            text = script.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        if "process_recording" in text:
            return script
    return None


def _write_new_pipeline_wrapper(wrapper_path: Path) -> None:
    """Create a tiny adapter that calls process_recording with tolerant kwargs.

    The upstream project has changed during development.  Instead of hard-coding one
    exact function signature, this wrapper inspects `process_recording()` and passes
    only parameters the installed version accepts.
    """
    wrapper_path.write_text(
        r'''
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path


project_root = Path(sys.argv[1]).resolve()
entry_script = Path(sys.argv[2]).resolve()
source = Path(sys.argv[3]).resolve()
work_root = Path(sys.argv[4]).resolve()
denoise_preset = sys.argv[5]
result_json = Path(sys.argv[6]).resolve()

sys.path.insert(0, str(project_root))
work_root.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("ee328_new_entry", entry_script)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import entry script: {entry_script}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# The teammate UI module imports evaluation defaults at module import time:
#     from evaluate_voiceprivacy import DEFAULT_ASV_PYTHON, ...
# Upstream values were authored on macOS/Linux and may point to paths such as
# /opt/anaconda3/..., which do not exist on this Windows fused project.  Override
# those imported module globals before calling process_recording(), while still
# respecting explicit environment variables if the user has configured model paths.
def _override_path_global(name: str, *env_names: str, fallback: object = None) -> None:
    raw = None
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            raw = value
            break
    if raw is None:
        raw = fallback if fallback is not None else getattr(module, name, "")
    try:
        setattr(module, name, Path(str(raw)).expanduser().resolve())
    except Exception:
        setattr(module, name, raw)

_override_path_global(
    "DEFAULT_ASV_PYTHON",
    "ASV_PYTHON",
    "VOICEPRIVACY_PYTHON",
    "PIPELINE_PYTHON",
    "VC_PYTHON",
    fallback=sys.executable,
)
_override_path_global(
    "DEFAULT_SPEAKER_MODEL",
    "VOICEPRIVACY_SPEAKER_MODEL",
    "SPEAKER_MODEL",
    fallback=getattr(module, "DEFAULT_SPEAKER_MODEL", ""),
)
_override_path_global(
    "DEFAULT_WHISPER_MODEL",
    "VOICEPRIVACY_WHISPER_MODEL",
    "WHISPER_MODEL",
    fallback=getattr(module, "DEFAULT_WHISPER_MODEL", ""),
)

if hasattr(module, "DEFAULT_ASV_PYTHON"):
    print(f"ℹ️ VoicePrivacy/ASV Python: {getattr(module, 'DEFAULT_ASV_PYTHON')}", flush=True)
if hasattr(module, "DEFAULT_SPEAKER_MODEL"):
    print(f"ℹ️ SpeechBrain speaker model: {getattr(module, 'DEFAULT_SPEAKER_MODEL')}", flush=True)
if hasattr(module, "DEFAULT_WHISPER_MODEL"):
    print(f"ℹ️ faster-whisper model: {getattr(module, 'DEFAULT_WHISPER_MODEL')}", flush=True)

fn = getattr(module, "process_recording", None)
if fn is None:
    raise RuntimeError(f"No process_recording() in {entry_script}")

sig = inspect.signature(fn)
params = sig.parameters
kwargs = {}
args = []

value_by_name = {
    "source": source,
    "source_path": source,
    "input": source,
    "input_path": source,
    "audio": source,
    "audio_path": source,
    "recording": source,
    "recording_path": source,
    "file_path": source,
    "wav_path": source,
    "project_root": project_root,
    "repo_root": project_root,
    "work_root": work_root,
    "work_dir": work_root,
    "output_root": work_root,
    "output_dir": work_root,
    "out_dir": work_root,
    "session_dir": work_root,
    "denoise_preset": denoise_preset,
    "noise_preset": denoise_preset,
    "target_gender": "male",
    "gender": "male",
}

if hasattr(module, "DemoConfig"):
    demo_config_cls = getattr(module, "DemoConfig")
    vc_python = Path(os.environ.get("VC_PYTHON") or getattr(module, "DEFAULT_VC_PYTHON", sys.executable)).expanduser().resolve()
    target_pool_config = Path(getattr(module, "DEFAULT_TARGET_POOL_CONFIG", project_root / "vc_target_pool_male_external.json")).expanduser().resolve()
    value_by_name["config"] = demo_config_cls(
        project_root=project_root,
        work_root=work_root,
        vc_python=vc_python,
        host="127.0.0.1",
        port=7860,
        target_pool_config=target_pool_config,
        max_targets=9,
    )

positional_source_used = False
for name, param in params.items():
    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
        continue
    if name in value_by_name:
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and not positional_source_used and name in {
            "source", "source_path", "input", "input_path", "audio", "audio_path", "recording", "recording_path", "file_path", "wav_path"
        }:
            kwargs[name] = value_by_name[name]
            positional_source_used = True
        else:
            kwargs[name] = value_by_name[name]

required = [
    p for p in params.values()
    if p.default is inspect._empty
    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    and p.name not in kwargs
]
fallback_values = [source, work_root, project_root, denoise_preset]
for p, value in zip(required, fallback_values):
    if p.kind == p.POSITIONAL_ONLY:
        args.append(value)
    else:
        kwargs[p.name] = value

print(f"📡 调用新版 process_recording: {entry_script.name}", flush=True)
print(f"ℹ️ kwargs={json.dumps(kwargs, ensure_ascii=False, default=str)}", flush=True)
result = fn(*args, **kwargs)

payload = {
    "entry_script": str(entry_script),
    "source": str(source),
    "work_root": str(work_root),
    "return_value": result,
}
result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"✅ 新版 process_recording 返回结果已写入: {result_json}", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
'''.lstrip(),
        encoding="utf-8",
    )


def _run_new_pipeline(project_root: Path, work_root: Path, source: Path, denoise_preset: str) -> dict[str, Any]:
    entry = _new_entry_script(project_root)
    if entry is None:
        raise FileNotFoundError("未检测到新版匿名化入口 run_privacy_optimized_recording.py / recording_demo_ui.py")

    work_root.mkdir(parents=True, exist_ok=True)
    wrapper = work_root / "_call_new_privacy_pipeline.py"
    wrapper_result = work_root / "_new_pipeline_return.json"
    _write_new_pipeline_wrapper(wrapper)

    pipeline_python = _resolve_pipeline_python()
    cmd = [
        str(pipeline_python),
        str(wrapper),
        str(project_root),
        str(entry),
        str(source),
        str(work_root),
        denoise_preset,
        str(wrapper_result),
    ]
    print(f"ℹ️ 新版匿名化入口 Python: {pipeline_python}", flush=True)
    print("📡 检测到新版匿名化入口，优先启动新版三方法流程:", " ".join(cmd), flush=True)
    _stream_subprocess(cmd, project_root, _external_env())

    return _find_new_pipeline_report(work_root, wrapper_result)


def _find_new_pipeline_report(work_root: Path, wrapper_result: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for name in ("demo_summary.json", "recording_summary.json", "privacy_summary.json"):
        candidates.extend(work_root.rglob(name))
    candidates = [p for p in candidates if p.exists()]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        report_path = candidates[0]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.setdefault("_report_path", str(report_path))
        return report

    if wrapper_result.exists():
        payload = json.loads(wrapper_result.read_text(encoding="utf-8"))
        returned = payload.get("return_value")
        if isinstance(returned, dict):
            returned.setdefault("_report_path", str(wrapper_result))
            return returned
        payload.setdefault("_report_path", str(wrapper_result))
        return payload

    raise FileNotFoundError(f"新版流程运行结束，但没有找到 demo_summary.json 或返回结果: {work_root}")


def _copy_new_variant_outputs(report: dict[str, Any], out_dir: Path, source: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    source_stem = _safe_stem(source)

    raw_results = report.get("results") or report.get("outputs") or report.get("variants") or []
    if isinstance(raw_results, dict):
        raw_results = [dict(value, method=key) if isinstance(value, dict) else {"method": key, "audio_path": value} for key, value in raw_results.items()]

    for idx, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        method = str(item.get("method") or item.get("variant") or item.get("name") or f"method_{idx + 1}")
        label = str(item.get("label") or item.get("display_name") or method)
        final_output = (
            item.get("audio_path")
            or item.get("output_path")
            or item.get("final_output")
            or item.get("path")
            or item.get("wav_path")
        )
        if not final_output:
            continue
        src_path = Path(str(final_output))
        if not src_path.is_absolute():
            report_path = report.get("_report_path")
            src_path = (Path(report_path).parent if report_path else Path.cwd()) / src_path
        if not src_path.exists():
            continue
        suffix = _safe_stem(Path(method))
        dest = out_dir / f"anon_{source_stem}_{suffix}.wav"
        shutil.copy2(src_path, dest)
        record = dict(item)
        record.update(
            {
                "variant": method,
                "label": label,
                "copied_output": str(dest),
                "source_pipeline_output": str(src_path),
            }
        )
        copied.append(record)

    priority = {"ppg_tone": 3, "metric_clarity": 2, "freevc_baseline": 1}

    def _rank(item: dict[str, Any]) -> tuple[float, float]:
        method = str(item.get("variant") or "")
        p = max((score for key, score in priority.items() if key in method), default=0)
        metric = item.get("score") or item.get("timbre_index") or item.get("privacy_score") or 0
        try:
            metric_f = float(metric)
        except Exception:
            metric_f = 0.0
        return float(p), metric_f

    best = max(copied, key=_rank) if copied else None
    return copied, best


def _run_best_available_pipeline(project_root: Path, work_root: Path, source: Path, denoise_preset: str) -> tuple[str, dict[str, Any], Path]:
    if _new_entry_script(project_root) is None:
        raise FileNotFoundError(
            "未检测到同学 Web UI 新版 process_recording 入口；本融合版要求只运行 "
            "FreeVC optimized male / Metric+clarity optimized male / PPG-tone optimized male 三方法流程。"
        )
    report = _run_new_pipeline(project_root, work_root, source, denoise_preset)
    report_path = Path(str(report.get("_report_path") or work_root / "demo_summary.json"))
    print("✅ 已使用同学 Web UI 新版三方法匿名化流程（无旧版回退）", flush=True)
    return "new", report, report_path


def _copy_variant_outputs(report: dict[str, Any], out_dir: Path, source: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    source_stem = _safe_stem(source)

    for variant_name, variant_data in (report.get("vc_variants") or {}).items():
        selections = variant_data.get("preferred_selections") or variant_data.get("selections") or []
        for item in selections:
            if not _variant_uses_real_vc(item):
                continue
            final_output = item.get("final_output") or item.get("selected_candidate")
            if not final_output:
                continue
            src_path = Path(final_output)
            if not src_path.exists():
                continue
            suffix = "male" if variant_name == "male_leaning" else str(variant_name)
            dest = out_dir / f"anon_{source_stem}_{suffix}.wav"
            shutil.copy2(src_path, dest)
            record = dict(item)
            record.update(
                {
                    "variant": variant_name,
                    "copied_output": str(dest),
                    "source_pipeline_output": str(src_path),
                }
            )
            copied.append(record)

    best = None
    if copied:
        best = max(copied, key=lambda item: float(item.get("score", float("-inf"))))
    return copied, best


def _variant_uses_real_vc(item: dict[str, Any]) -> bool:
    backend = str((item.get("profile") or {}).get("backend") or "").lower()
    return bool(backend) and "fallback" not in backend


def _assert_real_vc_outputs(variants: list[dict[str, Any]]) -> None:
    """Fail loudly if the pipeline only produced DSP fallback candidates.

    Fallback output is useful for classroom dry-runs, but in this fused GUI it is
    misleading because users expect actual speaker anonymization.  Raising here
    prevents copying/playing a near-original file as if it were anonymized.
    """
    if any(_variant_uses_real_vc(item) for item in variants):
        return

    diagnostic = []
    for item in variants[:4]:
        profile = item.get("profile") or {}
        diagnostic.append(str(profile.get("backend") or "unknown"))
    raise RuntimeError(
        "FreeVC/TTS 真正后端没有成功运行，pipeline 只生成了 DSP fallback 结果，"
        "因此听起来会接近原声。请确认 env_map.json 中的 `声线克隆（双音频）` "
        "或 `语音匿名化（FreeVC）` 指向已安装 Coqui TTS 的 Python 3.10 环境，"
        "并且首次运行时允许下载模型 voice_conversion_models/multilingual/vctk/freevc24。"
        f" 当前候选 backend={diagnostic}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge runner for EE328 speech anonymization pipeline.")
    parser.add_argument("--source", required=True, help="Input source audio file.")
    parser.add_argument(
        "--project-root",
        default=str(_default_project_root()),
        help="Path to EE328_Speech-Signal-Processing project.",
    )
    parser.add_argument("--work-root", default="", help="Directory for pipeline intermediate artifacts.")
    parser.add_argument("--out-dir", default="", help="Directory where final GUI-friendly wav files are copied.")
    parser.add_argument("--out-json", default="", help="Path to write compact runner result JSON.")
    parser.add_argument(
        "--denoise-preset",
        default="standard",
        choices=["light", "standard", "strong"],
        help="Denoise preset forwarded to the external pipeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"源音频不存在: {source}")
    if not project_root.exists():
        raise FileNotFoundError(f"匿名化模块目录不存在: {project_root}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_root = Path(args.work_root).expanduser().resolve() if args.work_root else _repo_root() / "_anon_work" / timestamp
    # Compatibility for GUI processes that were launched before this fix: they may
    # still pass a long work_root under the cloned teammate repo.  On Windows the
    # legacy pipeline creates deeply nested candidate filenames there and can hit
    # MAX_PATH, surfacing as FileNotFoundError from wavfile.write/open().
    try:
        if args.work_root and project_root in work_root.parents and "gui_work" in work_root.parts:
            short_root = _repo_root() / "_anon_work" / work_root.name
            print(f"ℹ️ 检测到过长匿名化工作目录，已改用短路径: {short_root}", flush=True)
            work_root = short_root.resolve()
    except Exception:
        pass
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else source.parent
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else out_dir / f"anon_{source.stem}_result.json"

    pipeline_mode, report, report_path = _run_best_available_pipeline(project_root, work_root, source, args.denoise_preset)

    if pipeline_mode == "legacy":
        _augment_report_with_voiceprivacy_metrics(project_root, work_root, source, report, report_path)

    if pipeline_mode == "new":
        variants, best = _copy_new_variant_outputs(report, out_dir, source)
    else:
        variants, best = _copy_variant_outputs(report, out_dir, source)
    if not variants:
        if pipeline_mode == "new":
            raise RuntimeError(
                "新版匿名化流程完成，但没有在 results/outputs/variants 中找到可复制的音频输出。"
                "请检查 demo_summary.json 里的 audio_path/output_path/final_output 字段。"
            )
        raise RuntimeError(
            "旧版 pipeline 完成，但没有找到真正 FreeVC 生成的男声匿名化结果。"
            "如果报告中 backend=freevc_fallback，说明当前环境没有成功调用 Coqui TTS/FreeVC。"
        )
    if pipeline_mode == "legacy":
        _assert_real_vc_outputs(variants)

    result = {
        "source": str(source),
        "project_root": str(project_root),
        "work_root": str(work_root),
        "pipeline_mode": pipeline_mode,
        "pipeline_report": str(report_path),
        "voiceprivacy_style_report": report.get("voiceprivacy_style_report"),
        "voiceprivacy_style_error": report.get("voiceprivacy_style_error"),
        "result_json": str(out_json),
        "variants": variants,
        "selected_output": best.get("copied_output") if best else None,
        "selected_variant": best.get("variant") if best else None,
        "selected_score": best.get("score") if best else None,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ 匿名化结果已整理完成", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"❌ 匿名化 runner 失败: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)