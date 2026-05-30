## **Codebase** 說明：

本專案欲建立 Novatek compiler 的模型轉換自動工具鏈，主要用於將 LLM（如 Qwen3-0.6B）轉換以得到 nvt_model.bin。Frontend 的 input 從 hugging face 端，輸出為 nvt_model.bin。以下說明：

### `qwen3-0.6B/` (convert tool)

- **`run_main_pipeline.py`**
自動化主程式。負責依序執行：環境設置、PyTorch ref_data generation、gen tool config、以及compiler 的呼叫指令。
- **`onnx2novaonnx_converter.py`**
這邊負責紀錄我們在 compiler 的 frontend 中作的修改。新增了 `pyop_rewrite()` 函式，負責將 onnx graph 的 Reciprocal node 依照文件 Novaic_Convertor_User_Guide_en.pdf 做 rewrite。
- **`node_rsqrt_pyop.py` & `cust_config.txt`**
PyOp  的實作與  cust config。
- **`gen_llm_reference_data_torch.py`**
透過 PyTorch 執行推論，並將模型的中間張量（如 Inputs Embeddings, Position IDs, Attention Mask, KV Cache）導出為 `.bin` 格式，作為 Reference Data。
- **`gen_gen_config.py`**
依據參數（Sequence Length, Head Dim 等），動態生成底層編譯器所需的 `gen_config.txt`，詳細定義了每一層的輸入/輸出維度與格式。

### `pyop_check/` (PyOp 獨立驗證)

- **`gen_onnx.py` & `gen_ref.py`**
用於生成一個僅包含 `Pow -> Reciprocal -> Add` 的精簡版 ONNX 模型與隨機輸入 `.bin` 檔，專門用來隔離並測試 PyOp fallback 的正確性。

### 使用方式

根據所選的 hugging face model，在相同資料夾下建立 `model.json` (可參考 repo)，轉換指令請執行 **`run_main_pipeline.py` ，**以下為具體指令 **(須根據 toolchain 位置修改 path)：**
```
python3 run_main_pipeline.py \
  --compiler-bin /path_to_toolchain/toolchain/closeprefix/bin/compiler.V30 \
  --config-dir /path_to_test-tutorial/test-tutorial/nvtai_tool
```
