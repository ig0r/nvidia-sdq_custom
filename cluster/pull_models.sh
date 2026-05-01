#!/usr/bin/env bash
set -u

OLLAMA_VERSION="${OLLAMA_VERSION:-0.22.0}"
OLLAMA_BIN="./${OLLAMA_VERSION}/bin/ollama"
MODELS_FILE="${1:-models.txt}"

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "Error: ollama binary not found or not executable at $OLLAMA_BIN" >&2
    exit 1
fi

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: models file not found: $MODELS_FILE" >&2
    exit 1
fi

total=0
ok=0
failed=()

while IFS= read -r line || [[ -n "$line" ]]; do
    # strip leading/trailing whitespace
    model="${line#"${line%%[![:space:]]*}"}"
    model="${model%"${model##*[![:space:]]}"}"

    # skip blanks and comments
    [[ -z "$model" || "$model" == \#* ]] && continue

    total=$((total + 1))
    echo "==> [$total] Pulling: $model"

    if "$OLLAMA_BIN" pull "$model"; then
        ok=$((ok + 1))
    else
        echo "!! Failed to pull: $model" >&2
        failed+=("$model")
    fi
done < "$MODELS_FILE"

echo
echo "Done: $ok/$total succeeded."
if (( ${#failed[@]} > 0 )); then
    echo "Failed models:"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi