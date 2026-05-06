import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PROMPTS = [
    "Give me a short introduction to large language models.",
    "Where in England was Dame Judi Dench born?",
    "From which country did Angola achieve independence in 1975?",
    "Which city does David Soul come from?",
    "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    "In which decade did Billboard magazine first publish and American hit chart?",
]

DEFAULT_COMPILER_ARGS = [
    "--chip",
    "690",
    "--bit-mode",
    "3",
    "--stripe-perfmode",
    "1",
    "--use-heteroge",
    "1",
    "--backend-order",
    "DSP_CPU",
    "--op-mode",
    "6",
    "--vlc-en",
    "1",
    "--openmp-en",
    "40",
    "--buf-thld",
    "65535",
    "--graph-thld",
    "65535",
    "--use-onnx",
    "--kvcache",
    "1",
    "--use-hf",
    "--hf-method",
    "2",
    "--model-processor",
    "2",
    "--verbose",
    "4",
    "--enc-len",
    "128",
    "--no-file-op",
    "0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full LLM pipeline for a list of models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models.json",
        help="Path to models.json",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: script directory)",
    )
    parser.add_argument(
        "--compiler-bin",
        type=str,
        default=None,
        help="Path to compiler.V30",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to nvtai_tool config root",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Fallback prompts file (one prompt per line)",
    )
    parser.add_argument(
        "--skip-compiler",
        action="store_true",
        help="Skip running compiler for all models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the whole pipeline on first error",
    )
    return parser.parse_args()


def load_models(models_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(models_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
        return data["models"]
    raise ValueError("models.json must be a list or contain a 'models' list.")


def sanitize_name(name: str) -> str:
    value = name.strip()
    value = re.sub(r"[\s.]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    if not value:
        raise ValueError("Unable to derive a valid model_name.")
    return value


def resolve_model_tag(entry: Dict[str, Any]) -> str:
    if "model_name" in entry:
        return sanitize_name(str(entry["model_name"]))
    model_id = str(entry.get("model_id", ""))
    if model_id:
        return sanitize_name(model_id.split("/")[-1])
    model_dir = entry.get("model_dir")
    if model_dir:
        return sanitize_name(Path(str(model_dir)).name)
    raise ValueError("model_name or model_id is required.")


def read_prompts_file(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def resolve_prompts(
    entry: Dict[str, Any],
    base_dir: Path,
    fallback_file: Optional[Path],
) -> List[str]:
    if "prompts" in entry:
        prompts = entry["prompts"]
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts must be a non-empty list.")
        return [str(prompt) for prompt in prompts]

    prompts_file = entry.get("prompts_file")
    if prompts_file:
        file_path = (base_dir / prompts_file).resolve()
        return read_prompts_file(file_path)

    if fallback_file:
        return read_prompts_file(fallback_file)

    return list(DEFAULT_PROMPTS)


def dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def ensure_hf_cli() -> None:
    if shutil.which("huggingface-cli") is None:
        raise RuntimeError("huggingface-cli is not available in PATH.")


def run_cmd(cmd: List[str], cwd: Optional[Path], dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def maybe_download_model(
    entry: Dict[str, Any],
    model_id: str,
    model_dir: Path,
    dry_run: bool,
) -> Path:
    download_mode = str(entry.get("download", "auto")).lower()
    revision = entry.get("revision")

    model_id_path = Path(model_id)
    if model_id_path.exists():
        return model_id_path

    if download_mode in {"false", "skip", "no"}:
        if not dir_has_files(model_dir):
            raise RuntimeError(f"model_dir does not exist: {model_dir}")
        return model_dir

    if download_mode not in {"auto", "true", "yes", "force"}:
        raise RuntimeError("download must be auto/true/false.")

    if download_mode == "auto" and dir_has_files(model_dir):
        return model_dir

    ensure_hf_cli()
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli",
        "download",
        model_id,
        "--local-dir",
        str(model_dir),
        "--local-dir-use-symlinks",
        "False",
    ]
    if revision:
        cmd.extend(["--revision", str(revision)])

    run_cmd(cmd, cwd=None, dry_run=dry_run)
    return model_dir


def get_attr(config: Any, names: List[str]) -> Optional[int]:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, int):
                return value
    return None


def resolve_model_params(
    entry: Dict[str, Any],
    model_source: Path,
    trust_remote_code: bool,
) -> Dict[str, int]:
    params: Dict[str, int] = {}
    overrides = {
        "hidden_size": entry.get("hidden_size"),
        "layer_count": entry.get("layer_count"),
        "attention_head": entry.get("attention_head"),
        "key_value_head": entry.get("key_value_head"),
        "head_dim": entry.get("head_dim"),
    }

    if all(value is not None for value in overrides.values()):
        return {key: int(value) for key, value in overrides.items()}

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to auto-resolve model params") from exc

    config = AutoConfig.from_pretrained(str(model_source), trust_remote_code=trust_remote_code)

    hidden_size = overrides["hidden_size"] or get_attr(config, ["hidden_size", "n_embd"])
    attention_head = overrides["attention_head"] or get_attr(
        config, ["num_attention_heads", "n_head"]
    )
    key_value_head = overrides["key_value_head"] or get_attr(
        config, ["num_key_value_heads"]
    )
    layer_count = overrides["layer_count"] or get_attr(
        config, ["num_hidden_layers", "n_layer", "num_layers"]
    )
    head_dim = overrides["head_dim"] or get_attr(config, ["head_dim"])

    if hidden_size is None or attention_head is None or layer_count is None:
        raise RuntimeError("Failed to resolve model params from config.")

    if key_value_head is None:
        key_value_head = attention_head

    if head_dim is None:
        head_dim = hidden_size // attention_head

    params["hidden_size"] = int(hidden_size)
    params["attention_head"] = int(attention_head)
    params["key_value_head"] = int(key_value_head)
    params["layer_count"] = int(layer_count)
    params["head_dim"] = int(head_dim)
    return params


def windows_rel(path: str) -> str:
    return path.replace("/", "\\")


def build_onnx_input_shape(entry: Dict[str, Any], seq_len: int, cache_len: int) -> str:
    shape_value = entry.get("onnx_input_shape")
    if shape_value is not None:
        if isinstance(shape_value, dict):
            return repr(shape_value)
        return str(shape_value)

    default_shape = {
        "input_ids": [1, seq_len],
        "past_key_values": [1, cache_len],
    }
    return repr(default_shape)


def run_model_pipeline(
    entry: Dict[str, Any],
    base_dir: Path,
    compiler_bin: Optional[Path],
    compiler_config_dir: Optional[Path],
    fallback_prompts_file: Optional[Path],
    skip_compiler: bool,
    dry_run: bool,
) -> None:
    model_id = str(entry.get("model_id", "")).strip()
    if not model_id:
        raise RuntimeError("model_id is required.")

    seq_len = int(entry.get("seq_len", 0))
    if seq_len <= 0:
        raise RuntimeError("seq_len must be > 0.")

    cache_len = int(entry.get("cache_len", seq_len))
    model_tag = resolve_model_tag(entry)
    data_prefix = sanitize_name(str(entry.get("data_prefix", model_tag)))

    model_dir = Path(entry.get("model_dir", base_dir / "nvtai_tool/input/model/customer" / model_tag))
    data_dir = Path(entry.get("data_dir", base_dir / "nvtai_tool/input/data" / model_tag))
    config_dir = Path(entry.get("config_dir", base_dir / "nvtai_tool/config" / model_tag / "cnn25"))

    prompts = resolve_prompts(entry, base_dir, fallback_prompts_file)
    ref_data_count = int(entry.get("ref_data_count", len(prompts)))

    trust_remote_code = bool(entry.get("trust_remote_code", False))

    model_source = maybe_download_model(entry, model_id, model_dir, dry_run)

    params = resolve_model_params(entry, model_source, trust_remote_code)

    if entry.get("export_embeddings"):
        embedding_out = entry.get("embedding_out", str(data_dir / f"{model_tag}_embedding.bin"))
        cmd = [
            sys.executable,
            str(base_dir / "export_llm_embeddings.py"),
            "--model",
            str(model_source),
            "--out-file",
            str(embedding_out),
        ]
        run_cmd(cmd, cwd=base_dir, dry_run=dry_run)

    data_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, start=1):
        out_dir = data_dir / f"{data_prefix}_seq{seq_len}_input{index}"
        cmd = [
            sys.executable,
            str(base_dir / "gen_llm_reference_data_torch.py"),
            "--model",
            str(model_source),
            "--prompt",
            prompt,
            "--max-seq-len",
            str(seq_len),
            "--out-dir",
            str(out_dir),
        ]
        if cache_len:
            cmd.extend(["--max-cache-len", str(cache_len)])
        run_cmd(cmd, cwd=base_dir, dry_run=dry_run)

    cmd = [
        sys.executable,
        str(base_dir / "gen_ref_list.py"),
        "--data-dir",
        str(data_dir),
        "--seq-len",
        str(seq_len),
        "--prefix",
        data_prefix,
    ]
    run_cmd(cmd, cwd=base_dir, dry_run=dry_run)

    config_dir.mkdir(parents=True, exist_ok=True)
    model_dir_config = entry.get(
        "model_dir_config",
        windows_rel(f"..\\nvtai_tool\\input\\model\\customer\\{model_tag}"),
    )
    ref_data_dir_config = entry.get(
        "ref_data_dir_config",
        windows_rel(f"..\\nvtai_tool\\input\\data\\{data_prefix}"),
    )

    cmd = [
        sys.executable,
        str(base_dir / "gen_gen_config.py"),
        "--path",
        model_tag,
        "--seq-len",
        str(seq_len),
        "--cache-len",
        str(cache_len),
        "--head-dim",
        str(params["head_dim"]),
        "--hidden-size",
        str(params["hidden_size"]),
        "--layer-count",
        str(params["layer_count"]),
        "--key-value-head",
        str(params["key_value_head"]),
        "--attention-head",
        str(params["attention_head"]),
        "--ref-data-count",
        str(ref_data_count),
        "--model-dir",
        model_dir_config,
        "--ref-data-dir",
        ref_data_dir_config,
        "--out-file",
        str(config_dir / "gen_config.txt"),
    ]
    run_cmd(cmd, cwd=base_dir, dry_run=dry_run)

    if skip_compiler or entry.get("skip_compiler"):
        return

    compiler_bin = Path(
        entry.get("compiler_bin", compiler_bin or base_dir / "../toolchain/closeprefix/bin/compiler.V30")
    )
    compiler_config_dir = Path(
        entry.get("compiler_config_dir", compiler_config_dir or base_dir / "nvtai_tool")
    )

    if not compiler_bin.exists():
        raise RuntimeError(f"Compiler not found: {compiler_bin}")

    pattern_name = sanitize_name(str(entry.get("pattern_name", model_tag)))
    onnx_shape = build_onnx_input_shape(entry, seq_len, cache_len)

    cmd = [
        str(compiler_bin),
        "--config-dir",
        str(compiler_config_dir),
        "--pattern-name",
        pattern_name,
        "--onnx-input-shape",
        onnx_shape,
        "--decoder-num",
        str(params["layer_count"]),
    ] + DEFAULT_COMPILER_ARGS

    run_cmd(cmd, cwd=base_dir, dry_run=dry_run)


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parent
    models_path = (base_dir / args.models).resolve()

    if not models_path.exists():
        print(f"models.json not found: {models_path}")
        return 1

    fallback_prompts_file = Path(args.prompts_file).resolve() if args.prompts_file else None

    models = load_models(models_path)
    if not models:
        print("No models found in models.json")
        return 1

    successes: List[str] = []
    failures: List[Tuple[str, str]] = []

    for entry in models:
        try:
            model_tag = resolve_model_tag(entry)
        except Exception as exc:
            failures.append(("<unknown>", str(exc)))
            if args.stop_on_error:
                break
            continue

        print(f"\n=== Running pipeline for {model_tag} ===")
        try:
            run_model_pipeline(
                entry=entry,
                base_dir=base_dir,
                compiler_bin=Path(args.compiler_bin) if args.compiler_bin else None,
                compiler_config_dir=Path(args.config_dir) if args.config_dir else None,
                fallback_prompts_file=fallback_prompts_file,
                skip_compiler=args.skip_compiler,
                dry_run=args.dry_run,
            )
            successes.append(model_tag)
        except Exception as exc:
            failures.append((model_tag, str(exc)))
            print(f"ERROR: {model_tag}: {exc}")
            if args.stop_on_error:
                break

    print("\n=== Pipeline Summary ===")
    print(f"Success: {len(successes)}")
    for name in successes:
        print(f"  - {name}")
    print(f"Failed: {len(failures)}")
    for name, reason in failures:
        print(f"  - {name}: {reason}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
