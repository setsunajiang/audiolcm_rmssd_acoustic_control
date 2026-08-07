# AudioLCM RMSSD Acoustic Control

This project develops and evaluates a low-latency AudioLCM-based audio
generation pipeline for continuous acoustic-feature control and future
RMSSD-guided biofeedback.

The project is being rebuilt from scratch. The original
`deap_rmssd_acoustic` project is treated as read-only reference material;
generated files and previous experiment outputs are not copied into this
repository.

## Current scope

Phase 1 establishes a reproducible AudioLCM baseline:

1. Set up a pinned Colab environment and AudioLCM source revision.
2. Load AudioLCM and its vocoder.
3. Generate AudioLCM's native audio output and derive the first five-second
   playback chunk.
4. Run one explicit warm-up generation outside the reported statistics.
5. Measure three warm batch-size-one generations against a five-second
   deadline.
6. Record the runtime configuration and benchmark results.

This phase evaluates generation infrastructure and latency only. It does not
claim continuous acoustic conditioning, RMSSD improvement, or a validated
real-time biofeedback loop.

## Project layout

```text
src/
  setup/
    requirements_colab.txt
    setup_audiolcm_colab.sh
  generation/
    benchmark_audiolcm.py
  features/
    acoustic_features.py
    extract_audio_features.py
  notebooks/
    audiolcm_colab_runner.ipynb
```

All maintained experiment code lives under `src/`. The Colab notebook is a
thin runner that installs dependencies and invokes the Python scripts; it is
not a second copy of the experiment implementation.

AudioLCM currently produces a nominal ten-second raw clip. The future
biofeedback loop will play and evaluate a five-second chunk from that output.
Benchmark reports therefore record both the actual generated duration and the
generation deadline separately.

The setup script downloads each required model asset directly into its final
location, verifies its expected byte size, and resumes partial downloads. A
repeated setup reuses the existing Python environment and completed assets.

## Phase 1 validated baseline

The Phase 1 benchmark was completed on August 7, 2026 with the following
configuration:

```text
GPU: NVIDIA A100-SXM4-40GB
AudioLCM revision: 51db10c49ee3e1a36938a0bd3791cb732165964a
PyTorch: 1.12.1+cu113
CUDA runtime: 11.3
Prompt: calm ambient piano, soft dynamics, no vocals
Warm-up runs: 1
Measured runs: 3
```

Measured results:

```text
Model and vocoder load: 29.743551 s
Warm-up generation: 1.823130 s
Measured generation 1: 0.226503 s
Measured generation 2: 0.222656 s
Measured generation 3: 0.224856 s
Mean measured generation: 0.224672 s
P95 measured generation: 0.226339 s
Mean generated duration: 9.984 s
Mean measured RTF: 0.022503
Five-second deadline success: 100%
Peak allocated GPU memory: 5.385517 GB
```

The benchmark times AudioLCM generation after the model and vocoder are
loaded. The first generation is reported as warm-up and excluded from the
measured mean, P95, RTF, and deadline-success statistics. Each native output
is approximately 9.984 seconds; the script also saves its first five seconds
as the future playback chunk.

This result validates low-latency generation on the measured A100 setup. It
does not validate direct control of the target acoustic features, continuous
feature conditioning, improved RMSSD, or an online biofeedback loop.

## Phase 2 acoustic features

Phase 2 uses one shared implementation for the six acoustic control features:

```text
tempo
rms_mean
zcr_mean
spectral_centroid_mean
spectral_contrast5_std
tonnetz1_mean
```

The extractor analyzes the first five seconds actually intended for playback.
It preserves the native sample rate, uses an FFT size of 2048 and hop length of
512, and explicitly uses five spectral-contrast bands. Five bands preserve the
required `spectral_contrast5_std` feature while remaining valid for AudioLCM's
16 kHz output.

## Research roadmap

1. Reproduce the low-latency AudioLCM baseline.
2. Implement one shared six-feature acoustic extractor.
3. Evaluate candidate reranking and explicit DSP control.
4. Connect the acoustic target interface to RMSSD optimization.
5. Train and evaluate continuous acoustic-feature conditioning.
6. Add chunk streaming and double-buffered playback.

The intended long-term control path is:

```text
RMSSD observation
  -> optimizer
  -> continuous acoustic target
  -> low-latency generator
  -> measured-feature validation
  -> next audio chunk
```
