import os
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def parser():
    parser = argparse.ArgumentParser(description="Python script to generate LLM reference data for Novaic Tool from Torch model.")

    parser.add_argument('--model', type=str, 
                        default="Qwen/Qwen3-0.6B",
                        help="Path to huggingface model folder")
    parser.add_argument('--prompt', type=str,
                        default="Give me a short introduction to large language models.",
                        help="Sample question for LLM.")
    parser.add_argument('--max-seq-len', type=int,
                        default=320,
                        help="Maximum sequence length for generated model.")
    parser.add_argument('--max-cache-len', type=int,
                        default=None,
                        help="Maximum cache length for generated model. (If not specified, default set to same as --max-seq-len)")
    parser.add_argument('--out-dir', type=str,
                        default="ref_data",
                        help="Output directory to store reference data")

    return parser.parse_args()

def gen_attention_map(token_len, max_seq_len, max_cache_len, kv_head=16):

    # Create attention with a look-ahead mask
    target_w = -np.ones((1, 1, token_len, token_len))
    target_w = np.triu(target_w, k=1)

    # Padded with additional sequence/cache length
    padded_target_w = -np.ones((1, 1, max_seq_len, max_seq_len+max_cache_len))
    padded_target_w[:, :, :token_len, max_cache_len:max_cache_len+token_len] = target_w
    
    # 20241231 added extended dimension for mask softmax purpose
    padded_target_w[:, :, token_len:, :] = np.repeat(padded_target_w[:, :, token_len-1, :][:, :, np.newaxis, :], repeats=(max_seq_len-token_len), axis=2)

    padded_target_w = np.repeat(padded_target_w, repeats=kv_head, axis=1)
    return padded_target_w

def attention_map_packbits(arr):

    if arr.ndim != 1:
        arr = arr.flatten()
    bits = (arr != 0).astype(np.uint8)
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.ones(pad, dtype=bits.dtype)])
    packed = np.packbits(bits, bitorder='little')
    return packed

if __name__ == "__main__":

    # Parse setting
    args = parser()
    MODEL = args.model
    PROMPT = args.prompt
    MAX_SEQ_LEN = args.max_seq_len
    MAX_CACHE_LEN = args.max_cache_len if args.max_cache_len is not None else MAX_SEQ_LEN
    OUTPUT_FOLDER = args.out_dir

    # load the tokenizer/model/embedding_table
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
     # if (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None) or not hasattr(tokenizer, 'chat_template'):
#         tokenizer.chat_template = """<|im_start|>system
# {system_message}
# <|im_end|>
# <|im_start|>user
# {user_message}
# <|im_end|>
# <|im_start|>assistant
# {assistant_message}
# <|im_end|>"""
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True).eval()
    input_embeddings = model.get_input_embeddings().weight
    eos_embeddings = input_embeddings[tokenizer.eos_token_id]

    # Load LLM setting
    NUM_ATTENTION_HEADS = model.config.num_attention_heads
    HIDDEN_SIZE = model.config.hidden_size
    NUM_KEY_VALUE_HEADS = model.config.num_key_value_heads
    HEAD_DIM = model.config.head_dim if hasattr(model.config, "head_dim") else HIDDEN_SIZE // NUM_ATTENTION_HEADS

    # PROMPT -> LLM input format
    messages = [{"role": "user", "content": PROMPT}]
    if (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None) or not hasattr(tokenizer, 'chat_template'):
        text = f'<|im_start|>{PROMPT}<|im_end|>'
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(model.device)
    input_len = len(input_ids[0])
    print(f'input_len: {input_len}')

    # Model Run
    outputs = model(input_ids)
    past_key_values = outputs.past_key_values # This contains the KV cache

    # inputs = tokenizer([text], return_tensors="pt")
    # text_outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)
    # text = tokenizer.decode(text_outputs[0], skip_special_tokens=True)
    # print()
    # print(f'Generated Output Text: `{text}`')
    # print()

    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save reference data output
    inputs_embeds = input_embeddings[input_ids[0]]  # Retrieve elements from a 2D PyTorch tensor based on given indices
    save_inputs_embeds = np.zeros((MAX_SEQ_LEN, HIDDEN_SIZE), dtype=np.float32)
    save_inputs_embeds[:input_len, :] = inputs_embeds.detach().cpu().to(torch.float32).numpy()
    save_inputs_embeds[input_len:, :] = eos_embeddings.detach().cpu().to(torch.float32).numpy()
    save_inputs_embeds.tofile(os.path.join(OUTPUT_FOLDER, "inputs_embeds.bin"))
    save_position_ids = np.arange(MAX_SEQ_LEN, dtype=np.uint16)
    save_position_ids.tofile(os.path.join(OUTPUT_FOLDER, "position_ids.bin"))
    save_attention_map = gen_attention_map(input_len, MAX_SEQ_LEN, MAX_CACHE_LEN, kv_head=NUM_ATTENTION_HEADS)
    bit_stream = attention_map_packbits(save_attention_map)
    bit_stream.tofile(os.path.join(OUTPUT_FOLDER, "attention_mask.bin"))
    padded_attn_map = np.bitwise_not(np.zeros(save_attention_map.size, dtype=np.uint8))
    padded_attn_map[:bit_stream.shape[0]] = bit_stream
    padded_attn_map.tofile(os.path.join(OUTPUT_FOLDER, "attention_mask_padded.bin"))
    # in case for newer transformer version
    for ind, (key, value) in enumerate(past_key_values):
        BATCH_SIZE = key.shape[0]
        out_past_key = np.zeros([BATCH_SIZE, NUM_KEY_VALUE_HEADS, MAX_CACHE_LEN, HEAD_DIM], dtype=np.float32)
        out_past_value = np.zeros([BATCH_SIZE, NUM_KEY_VALUE_HEADS, MAX_CACHE_LEN, HEAD_DIM], dtype=np.float32)
        out_past_key[:, :, :input_len, :] = key.detach().cpu().to(torch.float32).numpy()
        out_past_value[:, :, :input_len, :] = value.detach().cpu().to(torch.float32).numpy()
        out_past_key.tofile(os.path.join(OUTPUT_FOLDER, "past_key_values_{:}_key.bin".format(ind)))
        out_past_value.tofile(os.path.join(OUTPUT_FOLDER, "past_key_values_{:}_value.bin".format(ind)))
    print("Done!")
