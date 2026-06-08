#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate before/after F0 comparison histogram for test_16k.wav"""

from voice_gui import plot_f0_histogram

if __name__ == '__main__':
    input_wav = 'test_16k.wav'
    output_png = 'test_16k_f0_hist.png'
    print(f"Generating comparison histogram: {input_wav} -> {output_png}")
    plot_f0_histogram(input_wav, output_png)
    print(f"Done! Saved to {output_png}")
