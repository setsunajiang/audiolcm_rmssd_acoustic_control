import argparse
import csv
from pathlib import Path

from acoustic_features import extract_audio_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract six control features from played audio chunks."
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--playback-seconds", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    audio_paths = sorted(args.audio_dir.glob(args.pattern))

    if not audio_paths:
        raise ValueError(
            f"No audio files matching {args.pattern} in {args.audio_dir}"
        )

    rows = []
    for index, audio_path in enumerate(audio_paths, start=1):
        row = extract_audio_file(
            audio_path,
            playback_seconds=args.playback_seconds,
        )
        rows.append(row)
        print(
            f"{index}/{len(audio_paths)} {audio_path.name}: "
            f"tempo={row['tempo']:.3f}, "
            f"centroid={row['spectral_centroid_mean']:.3f}"
        )

    args.output_csv = args.output_csv.resolve()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows: {args.output_csv}")


if __name__ == "__main__":
    main()
