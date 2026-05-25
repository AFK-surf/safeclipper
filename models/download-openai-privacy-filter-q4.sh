#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/openai-privacy-filter"
ONNX_DIR="$MODEL_DIR/onnx"

mkdir -p "$ONNX_DIR"
cd "$MODEL_DIR"

for path in config.json tokenizer.json tokenizer_config.json onnx/model_q4.onnx onnx/model_q4.onnx_data; do
  if [[ ! -f "$path" ]]; then
    mkdir -p "$(dirname "$path")"
    /usr/bin/curl -L --fail --retry 3 \
      -o "$path" \
      "https://huggingface.co/openai/privacy-filter/resolve/main/$path"
  fi
done

if [[ ! -f "$ONNX_DIR/model_q4_embedded.onnx" ]]; then
  /opt/homebrew/bin/uv run --with onnx python - <<'PY'
from pathlib import Path
import onnx

onnx_dir = Path("onnx")
model = onnx.load(onnx_dir / "model_q4.onnx", load_external_data=True)
onnx.save_model(model, onnx_dir / "model_q4_embedded.onnx", save_as_external_data=False)
PY
fi

echo "$MODEL_DIR"
