"""
prepare_l3das22.py

Converts raw L3DAS22 Task1 data into a format usable alongside the
project's existing DeepFilterNet3 pipeline:

  - Each data point's predictor is actually stored as TWO separate
    4-channel files, `<base_name>_A.wav` and `<base_name>_B.wav`
    (one per mic array), not a single combined 8-channel file.
    This script extracts just the W (omnidirectional) channel from
    each of the A and B files and combines them into a single
    2-channel (WA, WB) pair - this matches a real 2-capsule dual-mic
    headset far better than the full 4-channel-per-array B-format data.
  - Resamples both the extracted 2-channel predictor audio and the
    mono clean target audio from L3DAS22's native 16kHz up to 48kHz,
    to match DeepFilterNet3's native sample rate.
  - Matches each `<base_name>_A.wav` / `<base_name>_B.wav` pair in
    `data/` to its single `<base_name>.wav` target file in `labels/`,
    and writes the converted (WA, WB) + clean pairs into a clean
    output structure.

USAGE
-----
    python data/prepare_l3das22.py \
        --input_root "Downloads/L3DAS22_Task1_train100/L3DAS22_Task1_train100" \
        --output_root "data/l3das22_converted"

Expects the input_root to contain two subfolders, exactly matching
the official L3DAS22 layout:
    <input_root>/data/    -> 8-channel predictor wav files (16kHz)
    <input_root>/labels/  -> mono clean target wav files (16kHz)

Produces:
    <output_root>/noisy/  -> 2-channel (WA, WB) wav files, 48kHz
    <output_root>/clean/  -> mono wav files, 48kHz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# The real on-disk layout splits each data point's predictor into TWO
# separate 4-channel files, not one combined 8-channel file:
#   <base_name>_A.wav -> [WA, YA, ZA, XA]
#   <base_name>_B.wav -> [WB, YB, ZB, XB]
# W (channel index 0 in each file) is the omnidirectional component -
# that's what we want for both A and B.
W_INDEX = 0

SOURCE_SR = 16000
TARGET_SR = 48000


def find_pairs(data_dir: Path, labels_dir: Path):
    """Match predictor _A/_B file pairs in data_dir to their single
    target file in labels_dir, by stripping the _A/_B suffix.
    Returns a list of (a_path, b_path, target_path) triples and
    reports any files that couldn't be matched.
    """
    a_files = {}
    b_files = {}
    for p in data_dir.glob("*.wav"):
        stem = p.stem  # filename without .wav
        if stem.endswith("_A"):
            a_files[stem[:-2]] = p
        elif stem.endswith("_B"):
            b_files[stem[:-2]] = p
        else:
            print(f"WARNING: predictor file '{p.name}' doesn't end in _A or _B, skipping.")

    label_files = {p.stem: p for p in labels_dir.glob("*.wav")}

    base_names_with_both = sorted(set(a_files) & set(b_files))
    base_names_missing_b = sorted(set(a_files) - set(b_files))
    base_names_missing_a = sorted(set(b_files) - set(a_files))

    if base_names_missing_b:
        print(f"WARNING: {len(base_names_missing_b)} base name(s) have an _A file "
              f"but no matching _B file, e.g.: {base_names_missing_b[:3]}")
    if base_names_missing_a:
        print(f"WARNING: {len(base_names_missing_a)} base name(s) have a _B file "
              f"but no matching _A file, e.g.: {base_names_missing_a[:3]}")

    matched_with_target = sorted(set(base_names_with_both) & set(label_files))
    missing_target = sorted(set(base_names_with_both) - set(label_files))
    if missing_target:
        print(f"WARNING: {len(missing_target)} A+B pair(s) have no matching target "
              f"file, e.g.: {missing_target[:3]}")

    triples = [(a_files[name], b_files[name], label_files[name]) for name in matched_with_target]
    return triples


def extract_w_channel(audio: np.ndarray) -> np.ndarray:
    """audio: shape (4, num_samples) or (num_samples, 4) - a single
    mic array's Ambisonic file. Returns shape (num_samples,) with
    just the W (omnidirectional) channel.
    """
    if audio.shape[0] == 4:
        return audio[W_INDEX]
    elif audio.shape[1] == 4:
        return audio[:, W_INDEX]
    else:
        raise ValueError(
            f"Expected a 4-channel predictor file, got shape {audio.shape}"
        )


def resample_channels(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """audio: shape (num_channels, num_samples) or (num_samples,) for mono.
    Resamples each channel independently.
    """
    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    resampled = [
        librosa.resample(audio[ch], orig_sr=orig_sr, target_sr=target_sr)
        for ch in range(audio.shape[0])
    ]
    return np.stack(resampled, axis=0)


def process_pair(a_path: Path, b_path: Path, target_path: Path,
                  noisy_out_dir: Path, clean_out_dir: Path):
    # --- predictor (noisy): load _A and _B separately, take W from each, -> 48kHz ---
    a_audio, sr_a = sf.read(str(a_path), always_2d=True)
    a_audio = a_audio.T  # (samples, channels) -> (channels, samples)
    b_audio, sr_b = sf.read(str(b_path), always_2d=True)
    b_audio = b_audio.T

    if sr_a != SOURCE_SR:
        print(f"NOTE: {a_path.name} has sample rate {sr_a}, expected {SOURCE_SR}")
    if sr_b != SOURCE_SR:
        print(f"NOTE: {b_path.name} has sample rate {sr_b}, expected {SOURCE_SR}")

    wa = extract_w_channel(a_audio)
    wb = extract_w_channel(b_audio)

    # Trim to equal length just in case A and B differ by a sample or two
    min_len = min(len(wa), len(wb))
    wa_wb = np.stack([wa[:min_len], wb[:min_len]], axis=0)  # (2, num_samples)

    wa_wb_48k = resample_channels(wa_wb, sr_a, TARGET_SR)

    # Use the target's base name for the output file, since A/B are now merged
    noisy_out_path = noisy_out_dir / target_path.name
    sf.write(str(noisy_out_path), wa_wb_48k.T, TARGET_SR)  # back to (samples, channels)

    # --- target (clean, mono -> 48kHz) ---
    target_audio, target_sr = sf.read(str(target_path), always_2d=False)
    if target_audio.ndim > 1:
        # in case it's not strictly mono, just take the first channel
        target_audio = target_audio[:, 0]
    target_48k = resample_channels(target_audio, target_sr, TARGET_SR)

    clean_out_path = clean_out_dir / target_path.name
    sf.write(str(clean_out_path), target_48k, TARGET_SR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True,
                         help="Path to the extracted L3DAS22 folder containing data/ and labels/")
    parser.add_argument("--output_root", type=str, default="data/l3das22_converted",
                         help="Where to write the converted noisy/ and clean/ folders")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process this many pairs (for a quick test run)")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    data_dir = input_root / "data"
    labels_dir = input_root / "labels"

    if not data_dir.is_dir() or not labels_dir.is_dir():
        print(f"ERROR: expected {data_dir} and {labels_dir} to both exist.")
        sys.exit(1)

    output_root = Path(args.output_root)
    noisy_out_dir = output_root / "noisy"
    clean_out_dir = output_root / "clean"
    noisy_out_dir.mkdir(parents=True, exist_ok=True)
    clean_out_dir.mkdir(parents=True, exist_ok=True)

    triples = find_pairs(data_dir, labels_dir)
    print(f"Found {len(triples)} matched A+B+target triples.")

    if args.limit:
        triples = triples[:args.limit]
        print(f"Limiting to first {len(triples)} triples for this run.")

    for i, (a_path, b_path, target_path) in enumerate(triples, start=1):
        try:
            process_pair(a_path, b_path, target_path, noisy_out_dir, clean_out_dir)
        except Exception as e:
            print(f"ERROR processing {target_path.name}: {e}")
            continue

        if i % 50 == 0 or i == len(triples):
            print(f"Processed {i}/{len(triples)}")

    print("Done.")
    print(f"Noisy (2ch, 48kHz) files written to: {noisy_out_dir}")
    print(f"Clean (mono, 48kHz) files written to: {clean_out_dir}")


if __name__ == "__main__":
    main()