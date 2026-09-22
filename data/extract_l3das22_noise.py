"""
extract_l3das22_noise.py

The official DeepFilterNet trainer only accepts separate speech-only,
noise-only, and RIR-only hdf5 pools - it mixes them together randomly
on the fly. L3DAS22 doesn't give us noise-only files though - only
already-mixed (speech+noise+real spatial capture) predictor files
plus their matching clean target.

This script approximates a noise-only signal for each L3DAS22 data
point by:
  1. Downmixing the 2-channel (WA, WB) predictor to mono, the same
     way as the real deployment pipeline (average, since these are
     genuinely correlated real captures much of the time - simple
     mean here, not the correlation-gated logic, since we just need
     a reasonable mono version of the mixture for subtraction).
  2. Subtracting the matching clean target from that downmixed
     mixture: noise_approx = mixture_mono - clean_mono

This will NOT be perfectly clean noise (the predictor's speech
component includes real room reverb/spatial coloration that the dry
clean target doesn't have, so some residual speech-shaped artifact
will leak into the "noise" here) - but it captures genuine real
acoustic noise *character* (from real recordings, not purely
synthetic mixes), which is the actual gap in the existing 63-clip
defence_noise_train pool. Treat this as a real, useful addition to
the noise pool, not a lab-perfect noise recording.

Input: the SAME data/prepare_l3das22.py raw source structure
    <input_root>/data/    -> <base>_A.wav / <base>_B.wav predictor files (16kHz)
    <input_root>/labels/  -> <base>.wav clean target files (16kHz)

Output: mono noise-only wav files at native 16kHz (matching the
sample rate your existing TRAIN_SET_SPEECH.hdf5 / TRAIN_SET_NOISE.hdf5
were built at, per your prepare_data.py --sr 16000 convention) -
no 48kHz resampling here, since these are headed for the same
Colab hdf5-building step as your original defence noise clips.

USAGE
-----
    python data/extract_l3das22_noise.py \
        --input_root "D:\Downloads\L3DAS22_Task1_train100\L3DAS22_Task1_train100" \
        --output_dir "data/l3das22_noise_extracted"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

W_INDEX = 0
SOURCE_SR = 16000


def find_triples(data_dir: Path, labels_dir: Path):
    a_files, b_files = {}, {}
    for p in data_dir.glob("*.wav"):
        stem = p.stem
        if stem.endswith("_A"):
            a_files[stem[:-2]] = p
        elif stem.endswith("_B"):
            b_files[stem[:-2]] = p

    label_files = {p.stem: p for p in labels_dir.glob("*.wav")}

    base_names = sorted(set(a_files) & set(b_files) & set(label_files))
    print(f"Found {len(base_names)} matched A+B+target triples.")
    return [(a_files[n], b_files[n], label_files[n]) for n in base_names]


def extract_w(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] == 4:
        return audio[W_INDEX]
    elif audio.shape[1] == 4:
        return audio[:, W_INDEX]
    raise ValueError(f"Expected 4-channel predictor file, got shape {audio.shape}")


def process_triple(a_path: Path, b_path: Path, target_path: Path, out_dir: Path):
    a_audio, sr_a = sf.read(str(a_path), always_2d=True)
    a_audio = a_audio.T
    b_audio, sr_b = sf.read(str(b_path), always_2d=True)
    b_audio = b_audio.T

    wa = extract_w(a_audio)
    wb = extract_w(b_audio)
    min_len = min(len(wa), len(wb))
    mixture_mono = (wa[:min_len] + wb[:min_len]) / 2.0  # simple downmix for subtraction

    clean_audio, sr_c = sf.read(str(target_path), dtype="float32")
    if clean_audio.ndim > 1:
        clean_audio = clean_audio[:, 0]

    n = min(len(mixture_mono), len(clean_audio))
    mixture_mono = mixture_mono[:n]
    clean_audio = clean_audio[:n]

    noise_approx = (mixture_mono - clean_audio).astype(np.float32)

    out_path = out_dir / f"{target_path.stem}_noise.wav"
    sf.write(str(out_path), noise_approx, sr_a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/l3das22_noise_extracted")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    data_dir = input_root / "data"
    labels_dir = input_root / "labels"

    if not data_dir.is_dir() or not labels_dir.is_dir():
        print(f"ERROR: expected {data_dir} and {labels_dir} to both exist.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = find_triples(data_dir, labels_dir)
    if args.limit:
        triples = triples[:args.limit]
        print(f"Limiting to first {len(triples)} for this run.")

    for i, (a_path, b_path, target_path) in enumerate(triples, start=1):
        try:
            process_triple(a_path, b_path, target_path, out_dir)
        except Exception as e:
            print(f"ERROR processing {target_path.name}: {e}")
            continue
        if i % 50 == 0 or i == len(triples):
            print(f"Processed {i}/{len(triples)}")

    print("Done.")
    print(f"Approximate noise-only files written to: {out_dir}")
    print("These are at native 16kHz, matching your existing prepare_data.py --sr 16000 convention.")


if __name__ == "__main__":
    main()