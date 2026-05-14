import onnx
from onnx import helper
from onnx import TensorProto

def create_test_onnx_model(output_path="test_model.onnx"):
    # 設定一個測試用的輸入 Shape
    # 這裡我刻意設定為 [1, 512, 1] 這樣即使你的 script fallback 也能一致，
    # 但因為我們會做 shape inference，你的腳本一定能精準抓到這個 shape。
    input_shape = [1, 512, 1]

    # 1. 定義模型的輸入 (X) 與輸出 (Y)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)

    # 2. 定義常數 (Initializers)
    # 用於平方的指數值 (2.0)
    exp_tensor = helper.make_tensor('exp_val', TensorProto.FLOAT, [], [2.0])
    # 用於最後加法的常數值 (例如加 1.0)
    add_tensor = helper.make_tensor('add_val', TensorProto.FLOAT, [], [1.0])

    # 3. 建立 Node
    # Node 1: 平方 (Pow) -> 對輸入 X 的所有元素做平方
    node_pow = helper.make_node(
        'Pow',
        inputs=['X', 'exp_val'],
        outputs=['pow_out'],
        name='Node_Pow'
    )

    # Node 2: 取倒數 (Reciprocal) -> 這是你要 fallback 到 cpu 的目標 Node
    node_reciprocal = helper.make_node(
        'Reciprocal',
        inputs=['pow_out'],
        outputs=['reciprocal_out'],
        name='node_rsqrt'
    )

    # Node 3: 統一做加法 (Add) -> 將取倒數的結果加上常數 1.0
    node_add = helper.make_node(
        'Add',
        inputs=['reciprocal_out', 'add_val'],
        outputs=['Y'],
        name='Node_Add'
    )

    # 4. 構建 Graph
    graph_def = helper.make_graph(
        nodes=[node_pow, node_reciprocal, node_add],
        name='PyOp_Fallback_Test_Graph',
        inputs=[X],
        outputs=[Y],
        initializer=[exp_tensor, add_tensor]
    )

    # 5. 構建 Model (指定 opset version，13 是目前相容性極高的主流版本)
    model_def = helper.make_model(graph_def, producer_name='gemini-test-builder')
    model_def.opset_import[0].version = 13

    # 6. 執行 Shape Inference (極度重要！)
    # 這一步會將中間層 'pow_out' 和 'reciprocal_out' 的 shape 資訊寫入 model.graph.value_info
    # 這樣你的 converter 中的 `for vi in model.graph.value_info:` 就能順利抓到 out_shape
    inferred_model = onnx.shape_inference.infer_shapes(model_def)

    # 7. 檢查模型合法性並儲存
    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, output_path)
    print(f"✅ 成功建立 ONNX 模型並儲存至: {output_path}")
    print("模型結構: X -> [Pow] -> [Reciprocal] -> [Add] -> Y")
    print("你可以將此模型餵給你的 onnx2novaonnx_converter.py 進行 PyOp 替換測試了。")

if __name__ == "__main__":
    create_test_onnx_model("test_fallback.onnx")