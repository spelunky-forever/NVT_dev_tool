import argparse
import re
from pathlib import Path
from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ref_list.txt files from LLM reference data folders."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing reference data input folders",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Sequence length used in list file names",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Input folder prefix (e.g. llama_3_2_1b)",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Override glob for input folders",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for list files (default: data-dir)",
    )
    return parser.parse_args()


def _input_sort_key(path: Path):
    match = re.search(r"input(\d+)$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def find_input_dirs(
    data_dir: Path,
    seq_len: int,
    prefix: Optional[str],
    input_glob: Optional[str],
):
    if input_glob:
        pattern = input_glob
    elif prefix:
        pattern = f"{prefix}_seq{seq_len}_input*"
    else:
        pattern = f"*_seq{seq_len}_input*"
    candidates = [path for path in data_dir.glob(pattern) if path.is_dir()]
    return sorted(candidates, key=_input_sort_key)


def detect_kv_indices(sample_dir: Path):
    key_indices = set()
    for file_path in sample_dir.glob("past_key_values_*_key.bin"):
        match = re.search(r"past_key_values_(\d+)_key\.bin$", file_path.name)
        if match:
            key_indices.add(int(match.group(1)))

    value_suffix_by_idx = {}
    for file_path in sample_dir.glob("past_key_values_*_value.bin"):
        match = re.search(r"past_key_values_(\d+)_value\.bin$", file_path.name)
        if match:
            value_suffix_by_idx[int(match.group(1))] = "value"
    for file_path in sample_dir.glob("past_key_values_*_values.bin"):
        match = re.search(r"past_key_values_(\d+)_values\.bin$", file_path.name)
        if match and int(match.group(1)) not in value_suffix_by_idx:
            value_suffix_by_idx[int(match.group(1))] = "values"

    indices = sorted(key_indices & set(value_suffix_by_idx))
    if not indices:
        raise ValueError("No past_key_values_{idx}_key.bin/value.bin files found.")
    return indices, value_suffix_by_idx


def resolve_attention_mask_name(sample_dir: Path):
    if (sample_dir / "attention_mask_padded.bin").exists():
        return "attention_mask_padded.bin"
    if (sample_dir / "attention_mask.bin").exists():
        return "attention_mask.bin"
    raise FileNotFoundError("Missing attention_mask_padded.bin or attention_mask.bin.")


def to_rel_posix(path: Path, base_dir: Path):
    return path.relative_to(base_dir).as_posix()


def write_list(out_dir: Path, filename: str, lines: List[str]):
    target = out_dir / filename
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = find_input_dirs(data_dir, args.seq_len, args.prefix, args.input_glob)
    if not input_dirs:
        raise FileNotFoundError("No input directories found for the given pattern.")

    sample_dir = input_dirs[0]
    kv_indices, value_suffix_by_idx = detect_kv_indices(sample_dir)
    attn_mask_name = resolve_attention_mask_name(sample_dir)

    inputs_embeds_lines = [
        to_rel_posix(path / "inputs_embeds.bin", data_dir) for path in input_dirs
    ]
    position_ids_lines = [
        to_rel_posix(path / "position_ids.bin", data_dir) for path in input_dirs
    ]
    attention_mask_lines = [
        to_rel_posix(path / attn_mask_name, data_dir) for path in input_dirs
    ]

    write_list(
        out_dir,
        f"ref_llm_inputs_embeds_seq{args.seq_len}_list.txt",
        inputs_embeds_lines,
    )
    write_list(
        out_dir,
        f"ref_llm_position_ids_seq{args.seq_len}_list.txt",
        position_ids_lines,
    )
    write_list(
        out_dir,
        f"ref_llm_attention_mask_seq{args.seq_len}_list.txt",
        attention_mask_lines,
    )

    for idx in kv_indices:
        key_lines = [
            to_rel_posix(path / f"past_key_values_{idx}_key.bin", data_dir)
            for path in input_dirs
        ]
        value_suffix = value_suffix_by_idx[idx]
        value_lines = [
            to_rel_posix(path / f"past_key_values_{idx}_{value_suffix}.bin", data_dir)
            for path in input_dirs
        ]

        write_list(
            out_dir,
            f"ref_llm_past_key_values_{idx}_key_seq{args.seq_len}_list.txt",
            key_lines,
        )
        write_list(
            out_dir,
            f"ref_llm_past_key_values_{idx}_values_seq{args.seq_len}_list.txt",
            value_lines,
        )

if __name__ == "__main__":
    main()
