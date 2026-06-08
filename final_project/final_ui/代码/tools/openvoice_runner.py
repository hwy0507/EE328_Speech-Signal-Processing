import argparse
import math
import os
import sys
import tempfile
import traceback


MIN_OPENVOICE_EMBED_SEC = 6.0


def _maybe_extend_audio_for_embedding(audio_path, min_duration_sec=MIN_OPENVOICE_EMBED_SEC):
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    min_duration_ms = int(float(min_duration_sec) * 1000)

    if duration_ms <= 0:
        raise RuntimeError(f"Audio is empty: {audio_path}")

    if duration_ms >= min_duration_ms:
        return audio_path, None, float(duration_ms) / 1000.0, float(duration_ms) / 1000.0

    repeat_count = max(2, int(math.ceil(min_duration_ms / float(duration_ms))))
    extended = AudioSegment.silent(duration=0)
    for _ in range(repeat_count):
        extended += audio

    fd, temp_path = tempfile.mkstemp(prefix="ov_embed_ext_", suffix=".wav")
    os.close(fd)
    extended.export(temp_path, format="wav")
    return (
        temp_path,
        temp_path,
        float(duration_ms) / 1000.0,
        float(len(extended)) / 1000.0,
    )

def main():
    parser = argparse.ArgumentParser(description="Run OpenVoice zero-shot voice conversion")
    parser.add_argument("--source", required=True, help="Source audio (wav recommended)")
    parser.add_argument("--target", required=True, help="Target reference audio")
    parser.add_argument("--out", required=True, help="Output wav path")
    parser.add_argument("--ckpt_dir", default="checkpoints_v2", help="OpenVoice checkpoints root")
    parser.add_argument("--device", default="auto", help="cpu/cuda/auto")
    parser.add_argument("--tau", type=float, default=0.3, help="OpenVoice tone color transfer strength")
    parser.add_argument("--no_vad", action="store_true", help="Disable VAD when extracting speaker embeddings")
    args = parser.parse_args()

    try:
        import torch
        import wavmark
        from openvoice.api import ToneColorConverter
        from openvoice import se_extractor
        from pydub import AudioSegment
    except Exception as e:
        print(f"ERROR: failed to import OpenVoice: {e}", file=sys.stderr)
        return 2

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt_dir = args.ckpt_dir
    converter_dir = os.path.join(ckpt_dir, "converter")
    config_path = os.path.join(converter_dir, "config.json")
    ckpt_path = os.path.join(converter_dir, "checkpoint.pth")

    ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ffmpeg", "bin", "ffmpeg.exe"))
    if os.path.exists(ffmpeg_path):
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        AudioSegment.converter = ffmpeg_path

    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        print(f"ERROR: OpenVoice checkpoints not found in {converter_dir}", file=sys.stderr)
        print("Expected files: config.json and checkpoint.pth", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    tau = max(0.0, min(1.0, float(args.tau)))
    use_vad = not args.no_vad

    try:
        # Avoid HuggingFace download for WavMark; we do not need watermarking.
        class _NoWatermark:
            def to(self, device):
                return self

        wavmark.load_model = lambda *args, **kwargs: _NoWatermark()

        tcc = ToneColorConverter(config_path, device=device)
        tcc.load_ckpt(ckpt_path)
        tcc.watermark_model = None

        source_for_se, source_temp, source_before, source_after = _maybe_extend_audio_for_embedding(args.source)
        target_for_se, target_temp, target_before, target_after = _maybe_extend_audio_for_embedding(args.target)

        if source_temp:
            print(
                f"INFO source audio auto-extended for embedding: {source_before:.2f}s -> {source_after:.2f}s"
            )
        if target_temp:
            print(
                f"INFO target audio auto-extended for embedding: {target_before:.2f}s -> {target_after:.2f}s"
            )

        # Extract embeddings with VAD enabled for cleaner timbre
        src_se, _ = se_extractor.get_se(source_for_se, tcc, target_dir=os.path.dirname(args.out), vad=use_vad)
        tgt_se, _ = se_extractor.get_se(target_for_se, tcc, target_dir=os.path.dirname(args.out), vad=use_vad)

        tcc.convert(
            audio_src_path=args.source,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=args.out,
            tau=tau,
            message="default",
        )
    except Exception as e:
        print(f"ERROR: OpenVoice conversion failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 4
    finally:
        for maybe_temp in [locals().get('source_temp'), locals().get('target_temp')]:
            try:
                if maybe_temp and os.path.exists(maybe_temp):
                    os.remove(maybe_temp)
            except Exception:
                pass

    print(f"INFO tau={tau:.2f}")
    print("DONE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
