"""
Runs every synthetic_defence_test noisy/clean pair through the ONNX
wrapper, reporting stereo channel correlation (for stereo files) next
to SNR improvement/STOI/PESQ for that same file - so low-correlation
files and their actual output quality can be compared directly, rather
than assumed from a single file (test4.wav). Mono files are included
too, correlation shown as N/A, so the full quality distribution is
visible for context, not just the flagged files.

Downmix logic mirrors src/audio/io.py's documented behavior (average if
correlation >= 0.95, otherwise pick the higher-RMS channel) so results
here reflect what live_demo.py would actually do, not eval_onnx_
wrapper.py's simpler mean(axis=1) downmix - the two can diverge on
stereo files, which is part of what this script is checking.

Read-only - does not modify any files.
"""

import glob
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo.live_demo import compute_snr_db
import src.df_onnx_dsp as wrap

NOISY_DIR = os.path.join(PROJECT_ROOT, "data", "l3das22_converted", "noisy")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "l3das22_converted", "clean")
ONNX_DIR = os.path.join(PROJECT_ROOT, "models", "onnx_export_v2")
CORRELATION_THRESHOLD = 0.95
MAX_FILES = 10


def downmix(data):
    if data.ndim == 1:
        return data, None
    if data.shape[1] < 2:
        return data[:, 0], None
    ch0 = data[:, 0]
    ch1 = data[:, 1]
    n = min(len(ch0), len(ch1))
    ch0 = ch0[:n]
    ch1 = ch1[:n]
    if np.std(ch0) == 0 or np.std(ch1) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(ch0, ch1)[0, 1])
    if correlation >= CORRELATION_THRESHOLD:
        mixed = (ch0 + ch1) / 2.0
    else:
        mixed = ch0 if np.sqrt(np.mean(ch0 ** 2)) >= np.sqrt(np.mean(ch1 ** 2)) else ch1
    return mixed.astype(np.float32), correlation


def evaluate_file(noisy_path, clean_path, enc_session, erb_dec_session, df_dec_session, erb_inv_fb):
    noisy_raw, sr_noisy = sf.read(noisy_path, dtype="float32")
    clean_raw, sr_clean = sf.read(clean_path, dtype="float32")

    noisy, correlation = downmix(noisy_raw)
    clean, _ = downmix(clean_raw)

    if sr_noisy != sr_clean:
        return {"file": os.path.basename(noisy_path), "error": "sample rate mismatch"}

    native_sr = sr_noisy
    if native_sr != wrap.SR:
        noisy_model_sr = librosa.resample(noisy, orig_sr=native_sr, target_sr=wrap.SR)
    else:
        noisy_model_sr = noisy

    try:
        enhanced_model_sr = wrap.enhance_chunk(noisy_model_sr, enc_session, erb_dec_session, df_dec_session, erb_inv_fb)
    except Exception as exc:
        return {"file": os.path.basename(noisy_path), "correlation": correlation, "error": str(exc)}

    if native_sr != wrap.SR:
        enhanced = librosa.resample(enhanced_model_sr, orig_sr=wrap.SR, target_sr=native_sr)
    else:
        enhanced = enhanced_model_sr

    n = min(len(clean), len(noisy), len(enhanced))
    clean = clean[:n]
    noisy = noisy[:n]
    enhanced = enhanced[:n]

    snr_before = compute_snr_db(clean, noisy)
    snr_after = compute_snr_db(clean, enhanced)
    snr_improvement = snr_after - snr_before

    try:
        stoi_value = float(stoi(clean, enhanced, native_sr, extended=False))
    except Exception:
        stoi_value = None

    try:
        pesq_sr = 16000  # pesq() only accepts exactly 8000 or 16000
        pesq_mode = "wb"
        clean_pesq = librosa.resample(clean, orig_sr=native_sr, target_sr=pesq_sr) if native_sr != pesq_sr else clean
        enhanced_pesq = librosa.resample(enhanced, orig_sr=native_sr, target_sr=pesq_sr) if native_sr != pesq_sr else enhanced
        pesq_value = float(pesq(pesq_sr, clean_pesq, enhanced_pesq, pesq_mode))
    except Exception:
        pesq_value = None

    return {
        "file": os.path.basename(noisy_path),
        "correlation": correlation,
        "snr_improvement_db": snr_improvement,
        "stoi": stoi_value,
        "pesq": pesq_value,
    }


def main():
    erb_inv_fb = wrap.build_erb_inv_fb(wrap.ERB_WIDTHS)
    enc_session, erb_dec_session, df_dec_session = wrap.load_sessions(ONNX_DIR)

    noisy_paths = sorted(glob.glob(os.path.join(NOISY_DIR, "*.wav")))[:MAX_FILES]
    if not noisy_paths:
        print(f"No wav files found in {NOISY_DIR}")
        return

    results = []
    for noisy_path in noisy_paths:
        filename = os.path.basename(noisy_path)
        clean_path = os.path.join(CLEAN_DIR, filename)
        if not os.path.exists(clean_path):
            continue
        print(f"Processing {filename}...")
        results.append(evaluate_file(noisy_path, clean_path, enc_session, erb_dec_session, df_dec_session, erb_inv_fb))

    print(f"\n{'=' * 90}")
    print(f"{'File':<20}{'Correlation':<14}{'SNR improve':<14}{'STOI':<10}{'PESQ':<10}")
    print(f"{'=' * 90}")
    for r in sorted(results, key=lambda r: (r.get("correlation") is None, r.get("correlation", 1.0))):
        if "error" in r:
            print(f"{r['file']:<20}{'ERROR: ' + r['error']}")
            continue
        corr_str = f"{r['correlation']:.3f}" if r["correlation"] is not None else "N/A (mono)"
        stoi_str = f"{r['stoi']:.4f}" if r["stoi"] is not None else "N/A"
        pesq_str = f"{r['pesq']:.3f}" if r["pesq"] is not None else "N/A"
        flag = " <-- LOW CORR" if r["correlation"] is not None and r["correlation"] < CORRELATION_THRESHOLD else ""
        print(f"{r['file']:<20}{corr_str:<14}{r['snr_improvement_db']:+.2f} dB{'':<6}{stoi_str:<10}{pesq_str:<10}{flag}")


if __name__ == "__main__":
    main()