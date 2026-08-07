#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${1:-/content/audiolcm_rmssd_acoustic_control}"
AUDIO_LCM_ROOT="${2:-/content/AudioLCM}"
ENV_ROOT="${3:-/content/audiolcm-env}"
ASSET_ROOT="${4:-/content/AudioLCM_assets}"
AUDIO_LCM_REVISION="51db10c49ee3e1a36938a0bd3791cb732165964a"
ASSET_REVISION="464dea17484a1f9808110704f7bf19b43b82e602"

if [[ ! -f "${PROJECT_ROOT}/src/setup/requirements_colab.txt" ]]; then
    echo "Project requirements not found under ${PROJECT_ROOT}." >&2
    exit 1
fi

python -m pip install -q uv

if [[ ! -d "${AUDIO_LCM_ROOT}/.git" ]]; then
    git clone https://github.com/Text-to-Audio/AudioLCM.git "${AUDIO_LCM_ROOT}"
fi

git -C "${AUDIO_LCM_ROOT}" checkout --detach "${AUDIO_LCM_REVISION}"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    uv venv --python 3.10 "${ENV_ROOT}"
fi

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

download_asset() {
    local expected_size="$1"
    local relative_path="$2"
    local output_path="${ASSET_ROOT}/${relative_path}"
    local download_url
    local actual_size=0

    download_url="https://huggingface.co/liuhuadai/AudioLCM/resolve/${ASSET_REVISION}/${relative_path}"
    mkdir -p "$(dirname "${output_path}")"

    if [[ -f "${output_path}" ]]; then
        actual_size="$(stat -c '%s' "${output_path}")"
    fi

    if [[ "${actual_size}" -gt "${expected_size}" ]]; then
        echo "Unexpected oversized asset: ${output_path}" >&2
        exit 1
    fi

    if [[ "${actual_size}" -lt "${expected_size}" ]]; then
        echo "Downloading ${relative_path} from byte ${actual_size}."
        curl \
            -L \
            --fail \
            --retry 3 \
            --retry-delay 5 \
            -C - \
            -o "${output_path}" \
            "${download_url}"
    fi

    actual_size="$(stat -c '%s' "${output_path}")"
    if [[ "${actual_size}" -ne "${expected_size}" ]]; then
        echo "Asset size mismatch: ${output_path}" >&2
        echo "Expected ${expected_size}, found ${actual_size}." >&2
        exit 1
    fi

    echo "Verified ${relative_path}: ${actual_size} bytes."
}

download_asset 5856422142 audiolcm.ckpt
download_asset 2588933243 useful_ckpt/AutoencoderKL/epoch=000032.ckpt
download_asset 2333972488 useful_ckpt/FrozenCLAPFLANEmbedder/CLAP_weights_2022.pth
download_asset 7308070914 useful_ckpt/LCM_audio/maa2.ckpt
download_asset 570 useful_ckpt/bert-base-uncased/config.json
download_asset 440473133 useful_ckpt/bert-base-uncased/pytorch_model.bin
download_asset 466062 useful_ckpt/bert-base-uncased/tokenizer.json
download_asset 48 useful_ckpt/bert-base-uncased/tokenizer_config.json
download_asset 231508 useful_ckpt/bert-base-uncased/vocab.txt
download_asset 607 useful_ckpt/t5-v1_1-large/config.json
download_asset 147 useful_ckpt/t5-v1_1-large/generation_config.json
download_asset 3132858253 useful_ckpt/t5-v1_1-large/pytorch_model.bin
download_asset 1786 useful_ckpt/t5-v1_1-large/special_tokens_map.json
download_asset 791656 useful_ckpt/t5-v1_1-large/spiece.model
download_asset 1857 useful_ckpt/t5-v1_1-large/tokenizer_config.json
download_asset 682 useful_ckpt/vocoder/args.yml
download_asset 449217313 useful_ckpt/vocoder/best_netG.pt

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
