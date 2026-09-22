"""
Scans data/processed/synthetic_defence_test/{noisy,clean} for any wav
files that are stereo, and reports the inter-channel correlation for
each one found. Flags any below the 0.95 threshold used by
src/audio/io.py's downmix - those are files where naive averaging
would risk partial phase cancellation, same root cause identified for
noisy/test4.wav. Read-only - does not modify or convert any files, just
reports findings so you know the real scope before deciding what to fix.
"""

import glob
import os
import sys

import numpy as np
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NOISY_DIR = os.path.join(PROJECT_ROOT, "data", "l3das22_converted", "noisy")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "l3das22_converted", "clean")
CORRELATION_THRESHOLD = 0.95


def check_file(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        return None
    if data.shape[1] < 2:
        return None
    ch0 = data[:, 0]
    ch1 = data[:, 1]
    n = min(len(ch0), len(ch1))
    if n == 0:
        return None
    ch0 = ch0[:n]
    ch1 = ch1[:n]
    if np.std(ch0) == 0 or np.std(ch1) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(ch0, ch1)[0, 1])
    return {
        "path": path,
        "channels": data.shape[1],
        "sample_rate": sr,
        "correlation": correlation,
        "below_threshold": correlation < CORRELATION_THRESHOLD,
    }


def scan_dir(label, directory):
    print(f"\nScanning {label}: {directory}")
    if not os.path.isdir(directory):
        print(f"  Directory not found, skipping.")
        return []

    paths = sorted(glob.glob(os.path.join(directory, "*.wav")))
    print(f"  {len(paths)} wav files found.")

    results = []
    for path in paths:
        result = check_file(path)
        if result is not None:
            results.append(result)
    return results


def main():
    all_results = []
    all_results += scan_dir("noisy", NOISY_DIR)
    all_results += scan_dir("clean", CLEAN_DIR)

    if not all_results:
        print("\nNo stereo files found in either folder - all files are mono.")
        return

    print(f"\n{'=' * 70}")
    print(f"STEREO FILES FOUND: {len(all_results)}")
    print(f"{'=' * 70}")

    flagged = [r for r in all_results if r["below_threshold"]]

    for result in all_results:
        flag = " <-- BELOW 0.95, phase cancellation risk" if result["below_threshold"] else ""
        print(f"{os.path.basename(result['path'])}: "
              f"correlation={result['correlation']:.3f}, "
              f"sr={result['sample_rate']}{flag}")

    print(f"\n{len(flagged)} of {len(all_results)} stereo files are below the "
          f"{CORRELATION_THRESHOLD} threshold.")
    if flagged:
        print("These files were likely phase-cancelled if evaluated with a plain "
              "mean(axis=1) downmix (e.g. eval_onnx_wrapper.py's current behavior).")


if __name__ == "__main__":
    main()