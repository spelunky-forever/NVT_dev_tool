import argparse
import os

HEADER_TEMPLATE = """
#######################################
#[PATH]
#######################################

### [GENERAL]
## model 
[path/model_dir] = {model_dir}

### [GENERATOR] 
## gen output root
[path/out_dir] = ..\\nvtai_tool\\output

#######################################
#[REFERENCE DATA]
#######################################

## img num [1, 1000]
[ref_data/num] = {ref_data_count}
[frontend/novaonnx/optim_level] = 1

#######################################
#[FUNCTION]
#######################################

### [MULTISCALE]
## kv cache gap
[multiscale/rank] = 3 # rank cnt
"""

PAST_KEY_TEMPLATE = """
[blob] = {idx}
[blob_name] = past_key_values.{cache_idx}.{type}
## img 
[path/ref_img_dir] = {ref_data_dir}
## list
[path/ref_list_path] = {ref_data_dir}\\ref_llm_past_key_values_{cache_idx}_{dir_type}_seq{seq_len}_list.txt

### [MULTISCALE]
## model mode
# 0: single-scaled net input, 1 nvt_model
# 1: multi-scaled net input, multiple nvt_model, sdk dynamically choose proper model to use
[multiscale/en] = 1

### [MULTISCALE]
## axis of kv cache rank
[multiscale/width] = {head_dim}, {head_dim}, {head_dim}
[multiscale/height] = {cache_len}, 510, 254
# modify

### [PREPROCESS]
## NUE2 input format

# 0: FMT_YONLY
# 1: FMT_RGB
# 2: FMT_YUV420
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/in/fmt] = 3

## input type for FMT_FEAT
# 0: TYPE_INT8
# 1: TYPE_UINT8
# 2: TYPE_INT16
# 3: TYPE_UINT16
# 6: TYPE_FLOAT32
[preproc/in/type] = 6

## input frac bit num for FMT_FEAT
[preproc/in/frac_bit_num] = 0

## input size[1, 4096]
# for [preproc/in/fmt] == "FMT_FEAT", this means input size
# for [preproc/in/fmt] != "FMT_FEAT", this input size is just an exmaple, actual input size will be dynamically modified in sdk
[preproc/in/width] = {head_dim} # modify
[preproc/in/height] = {cache_len}
[preproc/in/channel] = {key_value_head}
[preproc/in/batch] = 1
[preproc/in/time] = 1

# 0: RESIZE_WH
# 1: x
# 2: x
# 3: RESIZE_WH_WITH_TILING
[preproc/resize/mode] = 0

## RESIZE_WH PARAMETERS
## width & height [1, 1920]
#[preproc/resize/width] = 1
#[preproc/resize/height] = 1

## [MEAN SUB]
# 0: disable meansub
# 1: enable meansub
[preproc/meansub/en] = 0

## mean data mode
# 0: MEANSUB_DC
# 1: MEANSUB_PLANAR
[preproc/meansub/mode] = 0

## mean data format
# 0: TXT
# 1: BINARYPROTO
[preproc/meansub/fmt] = 0

## mean_path mean data fmt type
# 0: FMT_YONLY
# 1: FMT_RGB
# 2: x
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/meansub/fmt_type] = 0

## [NORMALIZE]
# 0: disable normalization
# 1: enable normalization
[preproc/normalize/en] = 0

## normalize scale
[preproc/normalize/scale] = 1.00000000

## [NUE2 OUT FORMAT]
# 0: PREPROC_OUT_FMT_YONLY
# 1: x
# 2: PREPROC_OUT_FMT_RGB
# 3: PREPROC_OUT_FMT_BGR
# 4: PREPROC_OUT_FMT_FEAT
[preproc/out_fmt] = 4
"""

INPUT_IDS_TEMPLATE = """
[blob] = {idx}
[blob_name] = input_ids
## img 
[path/ref_img_dir] = {ref_data_dir}
## list
[path/ref_list_path] = {ref_data_dir}\\ref_llm_inputs_embeds_seq{seq_len}_list.txt

### [MULTISCALE]
## model mode
# 0: single-scaled net input, 1 nvt_model
# 1: multi-scaled net input, multiple nvt_model, sdk dynamically choose proper model to use
[multiscale/en] = 1

### [MULTISCALE]
## axis of kv cache rank
[multiscale/width] = {hidden_size}, {hidden_size}, {hidden_size}  # modify
[multiscale/height] = {seq_len}, 2, 2

### [PREPROCESS]
## NUE2 input format

# 0: FMT_YONLY
# 1: FMT_RGB
# 2: FMT_YUV420
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/in/fmt] = 3

## input type for FMT_FEAT
# 0: TYPE_INT8
# 1: TYPE_UINT8
# 2: TYPE_INT16
# 3: TYPE_UINT16
# 6: TYPE_FLOAT32
[preproc/in/type] = 6

## input frac bit num for FMT_FEAT
[preproc/in/frac_bit_num] = 0

## input size[1, 4096]
# for [preproc/in/fmt] == "FMT_FEAT", this means input size
# for [preproc/in/fmt] != "FMT_FEAT", this input size is just an exmaple, actual input size will be dynamically modified in sdk
[preproc/in/width] = {hidden_size} # modify, hidden_size
[preproc/in/height] = {seq_len}
[preproc/in/channel] = 1
[preproc/in/batch] = 1
[preproc/in/time] = 1

# 0: RESIZE_WH
# 1: x
# 2: x
# 3: RESIZE_WH_WITH_TILING
[preproc/resize/mode] = 0

## RESIZE_WH PARAMETERS
## width & height [1, 1920]
#[preproc/resize/width] = 1
#[preproc/resize/height] = 1

## [MEAN SUB]
# 0: disable meansub
# 1: enable meansub
[preproc/meansub/en] = 0

## mean data mode
# 0: MEANSUB_DC
# 1: MEANSUB_PLANAR
[preproc/meansub/mode] = 0

## mean data format
# 0: TXT
# 1: BINARYPROTO
[preproc/meansub/fmt] = 0

## mean_path mean data fmt type
# 0: FMT_YONLY
# 1: FMT_RGB
# 2: x
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/meansub/fmt_type] = 3

## [NORMALIZE]
# 0: disable normalization
# 1: enable normalization
[preproc/normalize/en] = 0

## normalize scale
[preproc/normalize/scale] = 1.00000000

## [NUE2 OUT FORMAT]
# 0: PREPROC_OUT_FMT_YONLY
# 1: x
# 2: PREPROC_OUT_FMT_RGB
# 3: PREPROC_OUT_FMT_BGR
# 4: PREPROC_OUT_FMT_FEAT
[preproc/out_fmt] = 4
"""

POSITION_IDS_TEMPLATE = """
[blob] = {idx}
[blob_name] = position_ids
## img 
[path/ref_img_dir] = {ref_data_dir}
## list
[path/ref_list_path] = {ref_data_dir}\\ref_llm_position_ids_seq{seq_len}_list.txt

### [MULTISCALE]
## model mode
# 0: single-scaled net input, 1 nvt_model
# 1: multi-scaled net input, multiple nvt_model, sdk dynamically choose proper model to use
[multiscale/en] = 1

### [MULTISCALE]
## axis of kv cache rank
[multiscale/width] = {seq_len}, 2, 2 # seq_len
[multiscale/height] = 1, 1, 1

### [PREPROCESS]
## NUE2 input format

# 0: FMT_YONLY
# 1: FMT_RGB
# 2: FMT_YUV420
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/in/fmt] = 3

## input type for FMT_FEAT
# 0: TYPE_INT8
# 1: TYPE_UINT8
# 2: TYPE_INT16
# 3: TYPE_UINT16
# 6: TYPE_FLOAT32
[preproc/in/type] = 2

## input frac bit num for FMT_FEAT
[preproc/in/frac_bit_num] = 0

## input size[1, 4096]
# for [preproc/in/fmt] == "FMT_FEAT", this means input size
# for [preproc/in/fmt] != "FMT_FEAT", this input size is just an exmaple, actual input size will be dynamically modified in sdk
[preproc/in/width] = {seq_len} # seq_len
[preproc/in/height] = 1
[preproc/in/channel] = 1
[preproc/in/batch] = 1
[preproc/in/time] = 1

# 0: RESIZE_WH
# 1: x
# 2: x
# 3: RESIZE_WH_WITH_TILING
[preproc/resize/mode] = 0

## RESIZE_WH PARAMETERS
## width & height [1, 1920]
#[preproc/resize/width] = 1
#[preproc/resize/height] = 1

## [MEAN SUB]
# 0: disable meansub
# 1: enable meansub
[preproc/meansub/en] = 0

## mean data mode
# 0: MEANSUB_DC
# 1: MEANSUB_PLANAR
[preproc/meansub/mode] = 0

## mean data format
# 0: TXT
# 1: BINARYPROTO
[preproc/meansub/fmt] = 0

## mean_path mean data fmt type
# 0: FMT_YONLY
# 1: FMT_RGB
# 2: x
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/meansub/fmt_type] = 3

## [NORMALIZE]
# 0: disable normalization
# 1: enable normalization
[preproc/normalize/en] = 0

## normalize scale
[preproc/normalize/scale] = 1.00000000

## [NUE2 OUT FORMAT]
# 0: PREPROC_OUT_FMT_YONLY
# 1: x
# 2: PREPROC_OUT_FMT_RGB
# 3: PREPROC_OUT_FMT_BGR
# 4: PREPROC_OUT_FMT_FEAT
[preproc/out_fmt] = 4
"""

ATTENTION_MASK_TEMPLATE = """
[blob] = {idx}
[blob_name] = attention_mask
## img 
[path/ref_img_dir] = {ref_data_dir}
## list
[path/ref_list_path] = {ref_data_dir}\\ref_llm_attention_mask_seq{seq_len}_list.txt

### [MULTISCALE]
## model mode
# 0: single-scaled net input, 1 nvt_model
# 1: multi-scaled net input, multiple nvt_model, sdk dynamically choose proper model to use
[multiscale/en] = 1

### [MULTISCALE]
## axis of kv cache rank
[multiscale/width] = {input_len}, 512, 256
[multiscale/height] = {seq_len}, 2, 2

### [PREPROCESS]
## NUE2 input format

# 0: FMT_YONLY
# 1: FMT_RGB
# 2: FMT_YUV420
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/in/fmt] = 3

## input type for FMT_FEAT
# 0: TYPE_INT8
# 1: TYPE_UINT8
# 2: TYPE_INT16
# 3: TYPE_UINT16
# 6: TYPE_FLOAT32
# 7: TYPE_UINT1
[preproc/in/type] = 7

## input frac bit num for FMT_FEAT
[preproc/in/frac_bit_num] = 0

## input size[1, 4096]
# for [preproc/in/fmt] == "FMT_FEAT", this means input size
# for [preproc/in/fmt] != "FMT_FEAT", this input size is just an exmaple, actual input size will be dynamically modified in sdk
[preproc/in/width] = {input_len}
[preproc/in/height] = {seq_len}
[preproc/in/channel] = {attention_head}
[preproc/in/batch] = 1
[preproc/in/time] = 1

# 0: RESIZE_WH
# 1: x
# 2: x
# 3: RESIZE_WH_WITH_TILING
[preproc/resize/mode] = 0

## RESIZE_WH PARAMETERS
## width & height [1, 1920]
#[preproc/resize/width] = 512
#[preproc/resize/height] = 256

## [MEAN SUB]
# 0: disable meansub
# 1: enable meansub
[preproc/meansub/en] = 0

## mean data mode
# 0: MEANSUB_DC
# 1: MEANSUB_PLANAR
[preproc/meansub/mode] = 0

## mean data format
# 0: TXT
# 1: BINARYPROTO
[preproc/meansub/fmt] = 0

## mean_path mean data fmt type
# 0: FMT_YONLY
# 1: FMT_RGB
# 2: x
# 3: FMT_FEAT
# 4: FMT_BGR
# 5: x
[preproc/meansub/fmt_type] = 0

## [NORMALIZE]
# 0: disable normalization
# 1: enable normalization
[preproc/normalize/en] = 0

## normalize scale
[preproc/normalize/scale] = 1.00000000

## [NUE2 OUT FORMAT]
# 0: PREPROC_OUT_FMT_YONLY
# 1: x
# 2: PREPROC_OUT_FMT_RGB
# 3: PREPROC_OUT_FMT_BGR
# 4: PREPROC_OUT_FMT_FEAT
[preproc/out_fmt] = 4
"""

FOOTER_TEMPLATE = """
### [POSTPROCESS]
# 0: disable post process (classify accuracy)
# 1: enable post process (classify accuracy)
[postproc/en] = 0

#######################################
#[FEATURE PRECISION]
#######################################
### [INPUT PRECISION]
## 16bit input to functions, this priority is higher than [precision/mode]
# 0: disable 16bit
# 1: enable 16bit
[precision/in_hp/conv_en] = 0
[precision/in_hp/bnscale_en] = 0
[precision/in_hp/deconv_en] = 0
[precision/in_hp/fc_en] = 0
[precision/in_hp/eltwise_en] = 0
[precision/in_hp/roipool_en] = 0

### [BALANCE WEIGHT]
## outrange rate [1, 1023]
[precision/balance_weight/out_range_ratio] = 1000.000000

### [REF RELU OUTPUT]
# 0: current reference its layer out
# 1: current reference next relu out
[precision/ref_relu_outval_en] = 1

### [CROSS LAYER EQUALIZATION]
# 0: turn off CLE
# 1: turn on CLE
[precision/cle_en] = 0

[precision/cal_refine_iter] = 3

#######################################
#[WEIGHT COMPRESSION]
#######################################
# 0: disable quatization
# 1: enable quatization
[compression/method/quan_en] = 0

# 0: disable variable length coding
# 1: enable variable length coding
[compression/method/vlc_en] = 0

# 0: disable mixed compression
# 1: enable mixed compression
[compression/use_mixed_compression] = 0

#######################################
#[PERFORMANCE MODE]
#######################################

### [PERFORMANCE MODE]
# 0: APP_MODE
# 1: LINKED_LIST_SINGLE
[performance/mode] = 0

### [MEMORY MODE]
# 0: disable shrink memory
# 1: enable shrink memory
[performance/shrink_en] = 1

### [CNN ENGINE MODE] # only 525 can support CNN2
# 0: CNN1
# 1: CNN2
[performance/cnn_engine_mode] = 0

### [JOB SCHEDULING]
# 0: disable job scheduling
# 1: enable job scheduling
[performance/job_schedule_en] = 0
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate gen_config.txt for Novaic Tool LLM models."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="llama_3_2_1b",
        help="Model/data tag used in default paths",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override [path/model_dir] in config",
    )
    parser.add_argument(
        "--ref-data-dir",
        type=str,
        default=None,
        help="Override [path/ref_img_dir] and list path base",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--cache-len", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument("--key-value-head", type=int, default=8)
    parser.add_argument("--attention-head", type=int, default=32)
    parser.add_argument("--ref-data-count", type=int, default=6)
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Write output to file instead of stdout",
    )
    return parser.parse_args()


def build_config(args):
    seq_len = args.seq_len
    cache_len = args.cache_len if args.cache_len is not None else seq_len
    input_len = seq_len + cache_len

    model_dir = args.model_dir or f"..\\nvtai_tool\\input\\model\\customer\\{args.path}"
    ref_data_dir = args.ref_data_dir or f"..\\nvtai_tool\\input\\data\\{args.path}"

    parts = [
        HEADER_TEMPLATE.format(model_dir=model_dir, ref_data_count=args.ref_data_count),
        PAST_KEY_TEMPLATE.format(
            ref_data_dir=ref_data_dir,
            idx=0,
            cache_idx=0,
            seq_len=seq_len,
            type="key",
            dir_type="key",
            head_dim=args.head_dim,
            cache_len=cache_len,
            key_value_head=args.key_value_head,
        ),
        INPUT_IDS_TEMPLATE.format(
            ref_data_dir=ref_data_dir,
            idx=1,
            seq_len=seq_len,
            hidden_size=args.hidden_size,
        ),
        POSITION_IDS_TEMPLATE.format(ref_data_dir=ref_data_dir, idx=2, seq_len=seq_len),
        PAST_KEY_TEMPLATE.format(
            ref_data_dir=ref_data_dir,
            idx=3,
            cache_idx=0,
            seq_len=seq_len,
            type="value",
            dir_type="values",
            head_dim=args.head_dim,
            cache_len=cache_len,
            key_value_head=args.key_value_head,
        ),
    ]

    idx = 4
    cache_idx = 1
    for _ in range(args.layer_count - 1):
        parts.append(
            PAST_KEY_TEMPLATE.format(
                ref_data_dir=ref_data_dir,
                idx=idx,
                cache_idx=cache_idx,
                seq_len=seq_len,
                type="key",
                dir_type="key",
                head_dim=args.head_dim,
                cache_len=cache_len,
                key_value_head=args.key_value_head,
            )
        )
        parts.append(
            PAST_KEY_TEMPLATE.format(
                ref_data_dir=ref_data_dir,
                idx=idx + 1,
                cache_idx=cache_idx,
                seq_len=seq_len,
                type="value",
                dir_type="values",
                head_dim=args.head_dim,
                cache_len=cache_len,
                key_value_head=args.key_value_head,
            )
        )
        idx += 2
        cache_idx += 1

    assert args.layer_count * 2 + 2 == idx
    print(idx, args.layer_count * 2 + 2)
    parts.append(ATTENTION_MASK_TEMPLATE.format(
       ref_data_dir=ref_data_dir,
       idx=args.layer_count * 2 + 2,
       seq_len=seq_len,
       input_len=input_len,
       attention_head=args.attention_head,
    ))

    parts.append(FOOTER_TEMPLATE)
    return "".join(parts)


def main():
    args = parse_args()
    content = build_config(args)

    if args.out_file:
        out_dir = os.path.dirname(args.out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as file_handle:
            file_handle.write(content)
    else:
        print(content)


if __name__ == "__main__":
    main()
