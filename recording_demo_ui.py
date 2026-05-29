from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

from audio_preprocess import preprocess_file
from build_metric_attack_variants import ATTACK_PROFILES, apply_attack_profile
from ppg_tone_naturalizer import naturalize_file
from vc_candidate_builder import DEFAULT_VC_PYTHON, build_one_candidate

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = PROJECT_ROOT / "work_recording_demo"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
RECOMMENDED_TARGETS = {
    "male": PROJECT_ROOT.parent / "常规lab/lab9/s6.wav",
    "female": PROJECT_ROOT.parent / "常规lab/lab9/s4.wav",
}


@dataclass(frozen=True)
class DemoConfig:
    project_root: Path
    work_root: Path
    vc_python: Path
    host: str
    port: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a browser-based record-and-anonymize demo UI.")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT), help="Directory for recorded demo artifacts.")
    parser.add_argument("--vc-python", default=str(DEFAULT_VC_PYTHON), help="Python executable for FreeVC inference.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local Flask server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the local Flask server.")
    return parser.parse_args()


def safe_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"


def phone_clean_profile():
    for profile in ATTACK_PROFILES:
        if profile.name == "phone_clean":
            return profile
    raise RuntimeError("Missing phone_clean attack profile")


def profile_payload(target_name: str, target_reference: Path, postprocess: str) -> dict[str, Any]:
    return {
        "backend": "freevc",
        "name": f"freevc_demo_{target_name}",
        "target_reference": str(target_reference),
        "target_strategy": "single_ref_demo",
        "target_pool_name": f"demo_{target_name}",
        "target_reference_count": 1,
        "target_reference_paths": [str(target_reference)],
        "postprocess": postprocess,
    }


def build_public_url(session_id: str, path: Path, config: DemoConfig) -> str:
    relative = path.relative_to(config.work_root)
    return f"/outputs/{session_id}/{relative.as_posix().split('/', 1)[1]}"


def process_recording(input_path: Path, session_dir: Path, config: DemoConfig) -> dict[str, Any]:
    preprocess = preprocess_file(input_path, session_dir / "preprocess", denoise_preset="standard")
    denoised_path = preprocess.denoised_wav
    profile = phone_clean_profile()

    results: list[dict[str, Any]] = []
    for target_name, target_reference in RECOMMENDED_TARGETS.items():
        if not target_reference.exists():
            raise FileNotFoundError(f"Missing {target_name} target reference: {target_reference}")

        raw_output = session_dir / "raw_freevc" / f"demo_{target_name}_freevc.wav"
        raw_status = session_dir / "raw_freevc" / f"demo_{target_name}_freevc.json"
        vc_result = build_one_candidate(
            source_path=denoised_path,
            target_reference=target_reference,
            output_path=raw_output,
            status_json=raw_status,
            python_executable=config.vc_python,
        )
        if vc_result is None:
            raise RuntimeError(
                f"FreeVC failed for {target_name}. Check {raw_status.parent} and the speech-anon310 environment."
            )

        phone_output = session_dir / "metric_phone" / f"demo_{target_name}_metric_phone.wav"
        apply_attack_profile(raw_output, phone_output, profile)
        ppg_output = session_dir / "ppg_tone" / f"demo_{target_name}_ppg_tone.wav"
        ppg_metadata = session_dir / "ppg_tone" / f"demo_{target_name}_ppg_tone.json"
        naturalize_file(denoised_path, phone_output, ppg_output, ppg_metadata, strength=0.4)

        results.extend(
            [
                {
                    "label": f"Metric+phone {target_name}",
                    "method": "metric_phone",
                    "target": target_name,
                    "audio_url": build_public_url(session_dir.name, phone_output, config),
                    "audio_path": str(phone_output),
                    "profile": profile_payload(target_name, target_reference, "humanize_candidate + phone_clean"),
                },
                {
                    "label": f"PPG-tone {target_name}",
                    "method": "ppg_tone",
                    "target": target_name,
                    "audio_url": build_public_url(session_dir.name, ppg_output, config),
                    "audio_path": str(ppg_output),
                    "metadata_path": str(ppg_metadata),
                    "profile": profile_payload(
                        target_name,
                        target_reference,
                        "humanize_candidate + phone_clean + ppg_tone_naturalizer",
                    ),
                },
            ]
        )

    summary = {
        "session_id": session_dir.name,
        "input_path": str(input_path),
        "denoised_path": str(denoised_path),
        "denoised_url": build_public_url(session_dir.name, denoised_path, config),
        "results": results,
        "notes": [
            "This is a record-then-process demo, not low-latency streaming.",
            "The UI uses the same recommended target speakers as the report experiments: male=s6, female=s4.",
            "Metric+phone outputs prioritize ASV/ASR attack strength; PPG-tone adds Mandarin tone naturalization.",
        ],
    }
    (session_dir / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def create_app(config: DemoConfig) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.post("/api/anonymize")
    def anonymize():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file received."}), 400

        session_dir = config.work_root / safe_session_id()
        session_dir.mkdir(parents=True, exist_ok=True)
        uploaded = request.files["audio"]
        suffix = Path(uploaded.filename or "recording.webm").suffix or ".webm"
        input_path = session_dir / f"recording{suffix}"
        uploaded.save(input_path)

        try:
            summary = process_recording(input_path, session_dir, config)
        except Exception as exc:  # noqa: BLE001 - surface a clear UI error for the local demo.
            error_payload = {
                "session_id": session_dir.name,
                "error": str(exc),
                "input_path": str(input_path),
            }
            (session_dir / "error.json").write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return jsonify(error_payload), 500

        return jsonify(summary)

    @app.get("/outputs/<session_id>/<path:filename>")
    def outputs(session_id: str, filename: str):
        base = (config.work_root / session_id).resolve()
        target = (base / filename).resolve()
        if not target.is_file() or base not in target.parents:
            return jsonify({"error": "File not found"}), 404
        return send_file(target)

    @app.get("/api/config")
    def api_config():
        return jsonify(
            {
                "work_root": str(config.work_root),
                "vc_python": str(config.vc_python),
                "targets": {key: str(value) for key, value in RECOMMENDED_TARGETS.items()},
            }
        )

    return app


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Speech Anonymization Recorder</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #182026;
      --muted: #5b6770;
      --line: #d8dde3;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 26px;
      line-height: 1.2;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 16px 0;
    }
    button {
      appearance: none;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 650;
      cursor: pointer;
      min-height: 42px;
    }
    button:disabled {
      opacity: 0.48;
      cursor: not-allowed;
    }
    .primary { background: var(--accent); color: #fff; }
    .primary:hover:not(:disabled) { background: var(--accent-dark); }
    .secondary { background: #fff; border-color: var(--line); color: var(--text); }
    .danger { background: var(--danger); color: #fff; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 6px;
      background: #eef2f3;
      color: var(--muted);
      font-size: 14px;
    }
    .status.recording { color: var(--danger); background: #fff1f0; }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 99px;
      background: currentColor;
    }
    audio {
      width: 100%;
      margin-top: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .result h3 {
      margin: 0 0 4px;
      font-size: 16px;
    }
    .path {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .error {
      color: var(--danger);
      white-space: pre-wrap;
    }
    pre {
      background: #101820;
      color: #e8eef2;
      padding: 12px;
      border-radius: 8px;
      overflow: auto;
      font-size: 12px;
    }
    @media (max-width: 760px) {
      header { display: block; }
      .grid { grid-template-columns: 1fr; }
      main { padding: 20px 14px 32px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Speech Anonymization Recorder</h1>
        <p>点击开始录音，结束后自动生成 Metric+phone 和 PPG-tone 的男声/女声匿名化版本。</p>
      </div>
      <div class="status" id="status"><span class="dot"></span><span id="statusText">Ready</span></div>
    </header>

    <section class="panel">
      <h2>录音</h2>
      <p>建议录 3-10 秒。首次点击时浏览器会请求麦克风权限。</p>
      <div class="controls">
        <button class="primary" id="startBtn">开始录音</button>
        <button class="danger" id="stopBtn" disabled>结束录音</button>
        <button class="secondary" id="processBtn" disabled>匿名化处理</button>
      </div>
      <audio id="preview" controls hidden></audio>
      <div id="error" class="error"></div>
    </section>

    <section class="panel">
      <h2>输出结果</h2>
      <div id="results" class="grid"></div>
      <div id="summary"></div>
    </section>
  </main>

  <script>
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const processBtn = document.getElementById("processBtn");
    const preview = document.getElementById("preview");
    const statusBox = document.getElementById("status");
    const statusText = document.getElementById("statusText");
    const results = document.getElementById("results");
    const summary = document.getElementById("summary");
    const errorBox = document.getElementById("error");

    let mediaRecorder = null;
    let chunks = [];
    let recordedBlob = null;

    function setStatus(text, recording = false) {
      statusText.textContent = text;
      statusBox.classList.toggle("recording", recording);
    }

    function showError(message) {
      errorBox.textContent = message || "";
    }

    startBtn.addEventListener("click", async () => {
      showError("");
      results.innerHTML = "";
      summary.innerHTML = "";
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        chunks = [];
        recordedBlob = null;
        mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) chunks.push(event.data);
        };
        mediaRecorder.onstop = () => {
          recordedBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
          preview.src = URL.createObjectURL(recordedBlob);
          preview.hidden = false;
          processBtn.disabled = false;
          stream.getTracks().forEach(track => track.stop());
          setStatus("Recorded. Ready to anonymize.");
        };
        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        processBtn.disabled = true;
        setStatus("Recording", true);
      } catch (error) {
        showError("无法访问麦克风：" + error.message);
        setStatus("Microphone error");
      }
    });

    stopBtn.addEventListener("click", () => {
      if (!mediaRecorder) return;
      mediaRecorder.stop();
      startBtn.disabled = false;
      stopBtn.disabled = true;
    });

    processBtn.addEventListener("click", async () => {
      if (!recordedBlob) return;
      showError("");
      results.innerHTML = "";
      summary.innerHTML = "";
      processBtn.disabled = true;
      startBtn.disabled = true;
      setStatus("Processing");

      const formData = new FormData();
      const ext = recordedBlob.type.includes("wav") ? "wav" : "webm";
      formData.append("audio", recordedBlob, `recording.${ext}`);
      try {
        const response = await fetch("/api/anonymize", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || JSON.stringify(payload));
        }
        renderResults(payload);
        setStatus("Done");
      } catch (error) {
        showError("处理失败：\\n" + error.message);
        setStatus("Failed");
      } finally {
        processBtn.disabled = false;
        startBtn.disabled = false;
      }
    });

    function renderResults(payload) {
      results.innerHTML = "";
      for (const item of payload.results || []) {
        const card = document.createElement("div");
        card.className = "result";
        card.innerHTML = `
          <h3>${item.label}</h3>
          <p>${item.method === "ppg_tone" ? "中文声调自然化版本" : "指标增强版本"}</p>
          <audio controls src="${item.audio_url}"></audio>
          <div class="path">${item.audio_path}</div>
        `;
        results.appendChild(card);
      }
      summary.innerHTML = `
        <h3>Session</h3>
        <pre>${JSON.stringify({
          session_id: payload.session_id,
          denoised_path: payload.denoised_path,
          notes: payload.notes
        }, null, 2)}</pre>
      `;
    }
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    config = DemoConfig(
        project_root=PROJECT_ROOT,
        work_root=Path(args.work_root).expanduser().resolve(),
        vc_python=Path(args.vc_python).expanduser().resolve(),
        host=args.host,
        port=args.port,
    )
    config.work_root.mkdir(parents=True, exist_ok=True)
    app = create_app(config)
    print(json.dumps({key: str(value) for key, value in asdict(config).items()}, ensure_ascii=False, indent=2))
    app.run(host=config.host, port=config.port, debug=False)


if __name__ == "__main__":
    main()
