#!/bin/bash
download_huggingface() {
    # first wget with --no-clobber, then wget with --timestamping
    wget -nc "$1" -P "$2" || wget -N "$1" -P "$2"

    # Rename the downloaded file to the new specified filename
    local downloaded_file="${2}/$(basename $1)"
    if [ -e "$downloaded_file" ]; then
        mv "$downloaded_file" "${2}/$3"
    fi
}

TARGET_DIR="checkpoints"

download_huggingface 'https://civitai.com/api/download/models/1229708?type=Model&format=SafeTensor' "${TARGET_DIR}/checkpoints" "AnImageinXL40.safetensors"
