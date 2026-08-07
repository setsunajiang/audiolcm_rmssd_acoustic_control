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
