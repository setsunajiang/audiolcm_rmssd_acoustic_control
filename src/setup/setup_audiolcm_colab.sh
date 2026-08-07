#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${1:-/content/audiolcm_rmssd_acoustic_control}"
AUDIO_LCM_ROOT="${2:-/content/AudioLCM}"
ENV_ROOT="${3:-/content/audiolcm-env}"
ASSET_ROOT="${4:-/content/AudioLCM_assets}"

if [[ ! -f "${PROJECT_ROOT}/src/setup/requirements_colab.txt" ]]; then
    echo "Project requirements not found under ${PROJECT_ROOT}." >&2
    exit 1
fi

python -m pip install -q uv

if [[ ! -d "${AUDIO_LCM_ROOT}/.git" ]]; then
    git clone https://github.com/Text-to-Audio/AudioLCM.git "${AUDIO_LCM_ROOT}"
fi

uv venv --python 3.10 "${ENV_ROOT}"

uv pip install \
    --python "${ENV_ROOT}/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    "torch==1.12.1+cu113" \
    "torchaudio==0.12.1+cu113" \
    "torchvision==0.13.1+cu113"

uv pip install \
    --python "${ENV_ROOT}/bin/python" \
    --reinstall \
    "numpy==1.23.5"

uv pip install \
    --python "${ENV_ROOT}/bin/python" \
    -r "${PROJECT_ROOT}/src/setup/requirements_colab.txt"

uv pip install \
    --python "${ENV_ROOT}/bin/python" \
    --no-deps \
    "taming-transformers-rom1504==0.0.6"

"${ENV_ROOT}/bin/huggingface-cli" download liuhuadai/AudioLCM \
    audiolcm.ckpt \
    useful_ckpt/AutoencoderKL/epoch=000032.ckpt \
    useful_ckpt/FrozenCLAPFLANEmbedder/CLAP_weights_2022.pth \
    useful_ckpt/LCM_audio/maa2.ckpt \
    useful_ckpt/bert-base-uncased/config.json \
    useful_ckpt/bert-base-uncased/pytorch_model.bin \
    useful_ckpt/bert-base-uncased/tokenizer.json \
    useful_ckpt/bert-base-uncased/tokenizer_config.json \
    useful_ckpt/bert-base-uncased/vocab.txt \
    useful_ckpt/t5-v1_1-large/config.json \
    useful_ckpt/t5-v1_1-large/generation_config.json \
    useful_ckpt/t5-v1_1-large/pytorch_model.bin \
    useful_ckpt/t5-v1_1-large/special_tokens_map.json \
    useful_ckpt/t5-v1_1-large/spiece.model \
    useful_ckpt/t5-v1_1-large/tokenizer_config.json \
    useful_ckpt/vocoder/args.yml \
    useful_ckpt/vocoder/best_netG.pt \
    --local-dir "${ASSET_ROOT}" \
    --local-dir-use-symlinks False

mkdir -p \
    /content/ckpt \
    /content/logs/trainae/ckpt \
    /content/useful_ckpts/CLAP

ln -sfn \
    "${ASSET_ROOT}/useful_ckpt/LCM_audio/maa2.ckpt" \
    /content/ckpt/maa2.ckpt
ln -sfn \
    "${ASSET_ROOT}/useful_ckpt/AutoencoderKL/epoch=000032.ckpt" \
    /content/logs/trainae/ckpt/epoch=000032.ckpt
ln -sfn \
    "${ASSET_ROOT}/useful_ckpt/FrozenCLAPFLANEmbedder/CLAP_weights_2022.pth" \
    /content/useful_ckpts/CLAP/CLAP_weights_2022.pth
ln -sfn "${AUDIO_LCM_ROOT}/ldm" /content/ldm
ln -sfn \
    "${ASSET_ROOT}/useful_ckpt/t5-v1_1-large" \
    "${AUDIO_LCM_ROOT}/ldm/modules/encoders/CLAP/t5-v1_1-large"
ln -sfn \
    "${ASSET_ROOT}/useful_ckpt/bert-base-uncased" \
    "${AUDIO_LCM_ROOT}/ldm/modules/encoders/CLAP/bert-base-uncased"

echo "AudioLCM Colab environment is ready."
echo "Python: ${ENV_ROOT}/bin/python"
echo "AudioLCM: ${AUDIO_LCM_ROOT}"
echo "Assets: ${ASSET_ROOT}"
