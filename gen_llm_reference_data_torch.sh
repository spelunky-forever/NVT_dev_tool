#!/bin/bash
set -euo pipefail

usage() {
        cat <<EOF
Usage: $0 [options]

Options:
  --model <id_or_path>     Hugging Face model id or local model path
  --path <tag>             Output directory prefix (default: lfm2_5_350m)
  --max-seq-len <len>      Max sequence length (default: 512)
  --max-cache-len <len>    Max cache length (default: same as max-seq-len)
  --prompts-file <path>    File with one prompt per line
  -h, --help               Show this help
EOF
}

model="LiquidAI/LFM2.5-350M"
path="lfm2_5_350m"
max_seq_len=512
max_cache_len=""
prompts_file=""
prompts=(
"Give me a short introduction to large language models."
"Where in England was Dame Judi Dench born?"
"From which country did Angola achieve independence in 1975?"
"Which city does David Soul come from?"
"Which American-born Sinclair won the Nobel Prize for Literature in 1930?"
"In which decade did Billboard magazine first publish and American hit chart?"
)

while [[ $# -gt 0 ]]; do
        case "$1" in
                --model)
                        model="$2"
                        shift 2
                        ;;
                --path)
                        path="$2"
                        shift 2
                        ;;
                --max-seq-len)
                        max_seq_len="$2"
                        shift 2
                        ;;
                --max-cache-len)
                        max_cache_len="$2"
                        shift 2
                        ;;
                --prompts-file)
                        prompts_file="$2"
                        shift 2
                        ;;
                -h|--help)
                        usage
                        exit 0
                        ;;
                *)
                        echo "Unknown option: $1" >&2
                        usage
                        exit 1
                        ;;
        esac
done

if [[ -n "$prompts_file" ]]; then
        mapfile -t prompts < "$prompts_file"
fi

if [[ ${#prompts[@]} -eq 0 ]]; then
        echo "No prompts provided." >&2
        exit 1
fi

for i in "${!prompts[@]}"; do
        out_dir="${path}_seq${max_seq_len}_input$((i + 1))"
        cmd=(
                python3 gen_llm_reference_data_torch.py
                --model "$model"
                --prompt "${prompts[$i]}"
                --max-seq-len "$max_seq_len"
                --out-dir "$out_dir"
        )
        if [[ -n "$max_cache_len" ]]; then
                cmd+=(--max-cache-len "$max_cache_len")
        fi
        "${cmd[@]}"
done
