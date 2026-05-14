import argparse
import os
import torch
from transformers import AutoModelForCausalLM

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export LLM input embedding table to a binary file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LiquidAI/LFM2.5-350M",
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="lfm25_embedding.bin",
        help="Output file path for embedding table",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16", "auto"],
        help="Model load dtype",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model/tokenizer",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return DTYPE_MAP[dtype_name]


def main():
    args = parse_args()
    torch_dtype = resolve_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    ).eval()

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    input_embeddings = model.get_input_embeddings().weight
    input_embeddings.detach().cpu().numpy().tofile(args.out_file)


if __name__ == "__main__":
    main()
