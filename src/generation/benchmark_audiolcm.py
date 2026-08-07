import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf


DEFAULT_PROMPT = "calm ambient piano, soft dynamics, no vocals"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure warm batch-size-one AudioLCM generation latency."
    )
    parser.add_argument("--audiolcm-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--playback-seconds", type=float, default=5.0)
    parser.add_argument("--deadline-seconds", type=float, default=5.0)
    return parser.parse_args()


def synchronize():
    torch.cuda.synchronize()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_revision(repository):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")

    args.audiolcm_root = args.audiolcm_root.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(args.audiolcm_root)
    sys.path.insert(0, str(args.audiolcm_root))

    from ldm.models.diffusion.scheduling_lcm import LCMSampler
    from pythonscripts.InferAPI import GenSamples, load_model_from_config
    from vocoder.bigvgan.models import VocoderBigVGAN

    config_path = args.audiolcm_root / "configs/audiolcm.yaml"
    checkpoint_path = args.asset_root / "audiolcm.ckpt"
    vocoder_path = args.asset_root / "useful_ckpt/vocoder"
    raw_audio_dir = args.output_dir / "raw_audio"
    played_audio_dir = args.output_dir / "played_5s"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    played_audio_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    synchronize()
    load_started = time.perf_counter()

    model = load_model_from_config(
        config,
        str(checkpoint_path),
        verbose=False,
    )
    model = model.to("cuda")
    model.eval()
    sampler = LCMSampler(model)
    vocoder = VocoderBigVGAN(str(vocoder_path), torch.device("cuda"))
    generator = GenSamples(
        sampler=sampler,
        model=model,
        outpath=str(raw_audio_dir),
        vocoder=vocoder,
        save_mel=False,
        save_wav=True,
        original_inference_steps=config.model.params.num_ddim_timesteps,
    )

    synchronize()
    load_seconds = time.perf_counter() - load_started

    rows = []

    total_runs = args.warmup_runs + args.repeats

    for generation_index in range(1, total_runs + 1):
        is_warmup = generation_index <= args.warmup_runs
        phase = "warmup" if is_warmup else "measured"
        phase_index = (
            generation_index
            if is_warmup
            else generation_index - args.warmup_runs
        )
        output_name = f"{phase}_{phase_index:03d}"
        seed = args.base_seed + generation_index - 1
        set_seed(seed)

        prompt = {
            "ori_caption": args.prompt,
            "struct_caption": f"<{args.prompt}& all>",
        }

        synchronize()
        started = time.perf_counter()

        with torch.no_grad():
            with model.ema_scope():
                records = generator.gen_test_sample(
                    prompt,
                    wav_name=output_name,
                )

        synchronize()
        generation_seconds = time.perf_counter() - started

        audio_path = Path(records[0]["audio_path"]).resolve()
        audio_info = sf.info(audio_path)
        audio_seconds = audio_info.frames / audio_info.samplerate

        audio, sample_rate = sf.read(audio_path)
        playback_samples = int(args.playback_seconds * sample_rate)
        played_audio = audio[:playback_samples]
        played_audio_path = (
            played_audio_dir / f"{output_name}_first_5s.wav"
        )
        sf.write(
            played_audio_path,
            played_audio,
            sample_rate,
            subtype="PCM_16",
        )

        rows.append(
            {
                "generation_index": generation_index,
                "phase": phase,
                "phase_index": phase_index,
                "seed": seed,
                "prompt": args.prompt,
                "generation_seconds": generation_seconds,
                "audio_seconds": audio_seconds,
                "playback_seconds": len(played_audio) / sample_rate,
                "rtf": generation_seconds / audio_seconds,
                "deadline_seconds": args.deadline_seconds,
                "deadline_met": generation_seconds <= args.deadline_seconds,
                "audio_path": str(audio_path),
                "played_audio_path": str(played_audio_path.resolve()),
            }
        )

        print(
            f"{phase.title()} {phase_index}: "
            f"generation={generation_seconds:.6f}s, "
            f"audio={audio_seconds:.3f}s, "
            f"rtf={generation_seconds / audio_seconds:.6f}"
        )

    csv_path = args.output_dir / "audiolcm_latency_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    warmup_rows = [row for row in rows if row["phase"] == "warmup"]
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    measured_latencies = [
        row["generation_seconds"] for row in measured_rows
    ]
    summary = {
        "audiolcm_revision": git_revision(args.audiolcm_root),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "model_load_seconds": load_seconds,
        "warmup_runs": args.warmup_runs,
        "warmup_generation_seconds": [
            row["generation_seconds"] for row in warmup_rows
        ],
        "measured_repeats": args.repeats,
        "mean_generation_seconds": float(np.mean(measured_latencies)),
        "p95_generation_seconds": float(
            np.percentile(measured_latencies, 95)
        ),
        "mean_audio_seconds": float(
            np.mean([row["audio_seconds"] for row in measured_rows])
        ),
        "mean_rtf": float(
            np.mean([row["rtf"] for row in measured_rows])
        ),
        "deadline_seconds": args.deadline_seconds,
        "deadline_success_rate": float(
            np.mean([row["deadline_met"] for row in measured_rows])
        ),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "runs_csv": str(csv_path),
    }

    summary_path = args.output_dir / "audiolcm_latency_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Runs: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
