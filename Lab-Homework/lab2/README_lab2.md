# Speech Lab 2 MATLAB Scripts

Files in this folder:
- `problem1.m`: segmentation/labeling of `s5.wav`, zero-vowel and zero-consonant tests.
- `problem2.m`: record `/a i u/`, FFT spectra in dB, envelope peak formants, Praat comparison template.
- `problem3.m`: linear-phase FIR high-pass filtering for 60 Hz suppression, create new `test_16k.wav`.
- `problem1.mlx`, `problem2.mlx`, `problem3.mlx`: live script versions for direct run/check.
- `generate_mlx.m`: regenerate all `.mlx` files from `.m`.

## Run order
1. Open MATLAB in this folder.
2. Run `problem1`.
3. Run `problem2`.
4. Run `problem3`.

## Notes
- Each script currently points to:
  - `/Users/hwy/Desktop/个人/26春/语音信号处理/lab2`
- If your files move, edit `labDir` in the scripts.
- `problem2.m` includes a `praatFormants` placeholder matrix.
  - Fill values from Praat (row order: `IY`, `AA`, `UW`) and rerun for comparison table/plots.
