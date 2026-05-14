import onnx
#from onnx.tools import update_model_dims
import numpy as np
import json

import argparse
from opset_upgrade import *
from simplify_onnx_graph import *
from onnx_utility import *
from replace5dimExpandTo4dim import simplify_onnx_graph_after
from onnx2novaonnx_pipline import process_command,to_nova_onnx


def get_hf_input_shape(onnx_model):
    
    for input_tensor in onnx_model.graph.input:
        if input_tensor.name == "input_ids":
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["input_ids"] = shape

        if input_tensor.name == "attention_mask":
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["attention_mask"] = shape
        
        if input_tensor.name == "position_ids":
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["position_ids"] = shape
    
    Args().max_seq_len      = Args().input_shape["input_ids"][1]
    Args().max_cache_len    = Args().input_shape["input_ids"][1]
    Args().num_head         = Args().input_shape["attention_mask"][1]

    with open(os.path.join(Args().hf_input,"config.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
        Args().decoder_num = data["num_hidden_layers"]

    for input_tensor in onnx_model.graph.input:
        if "past_key_values" in input_tensor.name:
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["past_key_values"] = shape
            return 

def add_hw_mask_en(onnx_model):
    
    from hf2onnx.simplify_before_nova import add_atten_in
    onnx_model = add_atten_in(onnx_model)

    for idx in range(len(onnx_model.graph.node)):
        node = onnx_model.graph.node[idx]
        # if node.name in target_layers:
        if "attention_mask" in node.input and node.op_type == "Add":
            old_node = node
            hw_attr = helper.make_attribute('hw_mask_en', True)
            old_node.attribute.append(hw_attr)


def get_input_mean_kv(onnx_model):
    Args().decoder_num = 0
    for input_tensor in onnx_model.graph.input:
        if Args().input_mean.get(input_tensor.name,0) == 1:
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["input_ids"] = shape

        elif Args().input_mean.get(input_tensor.name,0) == 3:
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["attention_mask"] = shape
        
        elif Args().input_mean.get(input_tensor.name,0) == 2:
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["position_ids"] = shape
        
        else:
            Args().decoder_num += 1
    
    Args().max_seq_len      = Args().input_shape["input_ids"][1]
    Args().max_cache_len    = Args().input_shape["input_ids"][1]
    Args().num_head         = Args().input_shape["attention_mask"][1]

    for input_tensor in onnx_model.graph.input:
        if input_tensor.name not in Args().input_mean.keys():
            shape = []
            for d in input_tensor.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            Args().input_shape["past_key_values"] = shape
            return 

# for pyop
def pyop_rewrite(model):
    from onnx import helper
    pyop_count = 0
    
    for node in model.graph.node:
        if node.op_type == "Reciprocal":
            node.name = node.name + "_pyop"
            node.op_type = "PyOp"
            node.domain = "mydomain"

            out_tensor = node.output[0]
            out_shape = []
            for vi in model.graph.value_info:
                if vi.name == out_tensor:
                    out_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                    break
    
            if not out_shape:
                print("out_shape may have error")
                out_shape = [1, 512, 1]
                
            shape_attr = helper.make_attribute("shape", out_shape)
            node.attribute.append(shape_attr)
            
            module_name_str = "node_rsqrt_pyop"  # 檔名，不要加 .py
            class_name_str  = "RsqrtPyOp"     # 類別名
            module_attr = helper.make_attribute("module", module_name_str)
            class_attr = helper.make_attribute("class_name", class_name_str)
            
            node.attribute.append(module_attr)
            node.attribute.append(class_attr)
            
            pyop_count += 1
            print(f"[PyOp Injector] 成功將 {node.name} 替換為 PyOp, 提取 Shape: {out_shape}")
            break
            
    if pyop_count > 0:
        opset = model.opset_import.add()
        opset.domain = "mydomain"
        opset.version = 1
        print(f"完美注入 總共將 {pyop_count} 個 Reciprocal 替換成了 PyOp")
    else:
        print("No rewrite implemented")

    return model

    

if __name__ == '__main__':
    args = process_command()
    if args.no_file_op == 1:
        onnx_model = loadonnxmodel_shm(args.input_model_size, args.name, args.verbose)
    elif args.no_file_op == 2:
        onnx_model = loadonnxmodel_shm2(args.input_model_size, args.name, args.verbose)
    else:
        onnx_model = onnx.load(args.input)
        
        print("input:", args.input)
        print("output:", args.output)

    if Args().hf2onnx:
        get_hf_input_shape(onnx_model)      # get args for hf-flow
    elif not Args().hf2onnx and Args().kvcache==1 and Args().input_mean != {}:
        get_input_mean_kv(onnx_model)       # get args for llm-kv-cache wihtout hf

    onnx_model = to_nova_onnx(onnx_model, args)

    if Args().hf2onnx:
        add_hw_mask_en(onnx_model)

    # for pyop
    onnx_model = pyop_rewrite(onnx_model)
    
    save_onnx_model(onnx_model, args.no_file_op, args.output, args.name, args.verbose)
