import onnx
from onnx import helper
import sys

def modify_onnx_model(input_onnx, output_onnx):
    # 1. 載入模型結構 (不載入外部權重以節省記憶體與避免報錯)
    print(f"正在載入模型: {input_onnx}")
    model = onnx.load(input_onnx, load_external_data=False)

    # 2. 註冊 PyOp 專用的自訂網域 (domain)
    opset = model.opset_import.add()
    opset.domain = "mydomain"
    opset.version = 1

    target_name = "node_rsqrt" # 我們要替換的目標算子
    target_idx = -1
    target_node = None

    # 3. 尋找目標 Node
    for i, node in enumerate(model.graph.node):
        if node.name == target_name:
            target_node = node
            target_idx = i
            break

    if target_node is None:
        print(f"找不到節點: {target_name}")
        return

    print(f"找到目標節點: {target_node.name} (OpType: {target_node.op_type})")

    # 4. 建立新的 PyOp Node，繼承原本的 input 與 output
    pyop_node = helper.make_node(
        op_type="PyOp",
        inputs=target_node.input,
        outputs=target_node.output,
        name=target_node.name + "_pyop",
        domain="mydomain",
        # 這裡設定的 module 和 class_name 會被 C++ 底層讀取
        module="my_custom_cpu_op",  # 對應下一個步驟的 Python 檔名
        class_name="MyReciprocal"   # 對應下一個步驟的 Class 名稱
    )

    # 5. 替換原本的節點
    model.graph.node.remove(target_node)
    model.graph.node.insert(target_idx, pyop_node)

    # 6. 儲存修改後的模型
    onnx.save(model, output_onnx)
    print(f"改圖成功！已另存為: {output_onnx}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        modify_onnx_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 modify_onnx.py <input.onnx> <output.onnx>")