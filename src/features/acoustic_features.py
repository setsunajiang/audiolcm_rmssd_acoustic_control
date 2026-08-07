from pathlib import Path

import librosa
import numpy as np


CONTROL_FEATURES = (
    "tempo",
    "rms_mean",
    "zcr_mean",
    "spectral_centroid_mean",
    "spectral_contrast5_std",
    "tonnetz1_mean",
)

HOP_LENGTH = 512
N_FFT = 2048
SPECTRAL_CONTRAST_BANDS = 5


def extract_control_features(audio, sample_rate):
    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_bands=SPECTRAL_CONTRAST_BANDS,
    )
    harmonic_audio = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(
        y=harmonic_audio,
        sr=sample_rate,
    )

    return {
        "tempo": tempo,
        "rms_mean": float(np.mean(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_contrast5_std": float(np.std(contrast[4])),
        "tonnetz1_mean": float(np.mean(tonnetz[0])),
    }


def extract_audio_file(audio_path, playback_seconds=5.0):
    audio_path = Path(audio_path).resolve()
    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    source_duration_seconds = len(audio) / sample_rate
    playback_samples = int(playback_seconds * sample_rate)
    played_audio = audio[:playback_samples]
    analyzed_duration_seconds = len(played_audio) / sample_rate

    features = extract_control_features(
        played_audio,
        sample_rate,
    )

    return {
        "audio_file": audio_path.name,
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "source_duration_seconds": source_duration_seconds,
        "analyzed_duration_seconds": analyzed_duration_seconds,
        **features,
    }
