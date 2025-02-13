download_huggingface() {
    # first wget with --no-clobber, then wget with --timestamping
    wget -nc "$1" -P "$2" || wget -N "$1" -P "$2"
    # wget --header="Authorization: Bearer ${HF_TOKEN}" -nc "$1" -P "$2" || wget --header="Authorization: Bearer ${HF_TOKEN}" -N "$1" -P "$2"
}

TARGET_DIR="checkpoints"

download_huggingface download_huggingface 'https://civitai.com/api/download/models/1408658?type=Model&format=SafeTensor&size=full&fp=fp16' "${TARGET_DIR}/checkpoints"