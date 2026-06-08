import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run voice conversion via Coqui TTS from external env')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    try:
        from TTS.api import TTS
    except Exception as e:
        print(f"ERROR: failed to import TTS: {e}", file=sys.stderr)
        sys.exit(2)

    model_name = 'voice_conversion_models/multilingual/vctk/freevc24'
    print('Loading model:', model_name)
    try:
        tts = TTS(model_name)
    except Exception as e:
        print(f"ERROR: failed to load model: {e}", file=sys.stderr)
        sys.exit(3)

    print('Running voice conversion...')
    try:
        tts.voice_conversion_to_file(source_wav=args.source, target_wav=args.target, file_path=args.out)
    except Exception as e:
        print(f"ERROR: voice conversion failed: {e}", file=sys.stderr)
        sys.exit(4)

    print('DONE', args.out)
    return 0

if __name__ == '__main__':
    sys.exit(main())
