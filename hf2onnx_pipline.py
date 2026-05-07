# from hf2onnx_utility import *
import argparse
from onnx import helper
from hf2onnx.hf2onnx_utility import *
from hf2onnx.simplify_before_nova import *

def str_to_dict(str_dict):
    input_shape_dict = None
    try:
        input_shape_dict = eval(str_dict)
    except:
        assert isinstance(input_shape_dict, dict), "Please make sure --input_shape or -is is a dict-like string. For example: \"{\"inputA\":[1,3,640,320]}\". "
    return input_shape_dict

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',  type=str, required=True,  help='input model path')
    parser.add_argument('--output', '-o', type=str, required=True, help='output model path')    
    parser.add_argument('--method', '-m', type=str, default = "Model2Onnx_Auto", help='How to export onnx files')    
    parser.add_argument('--model_processor', '-mp', type=str, default = "Default", help='model_processor')    # TODO help
    parser.add_argument('-is', '--input_shape', type=str_to_dict, default = {}, help='Set onnx model input shape, like "{\'inputA\':[1,3,640,320]}"')
    
    return parser.parse_args()

def onnx_datatype_to_npType(data_type):
    if data_type == 1:     
        return np.float32
    elif data_type == 2:   
        return np.uint8
    elif data_type == 3:   
        return np.int8
    elif data_type == 4:
        return np.uint16
    elif data_type == 5:
        return np.int16
    elif data_type == 6:  
        return np.int32
    elif data_type == 7: 
        return np.int64
    elif data_type == 8:  # string
        return str
    elif data_type == 9:
        return bool
    elif data_type == 10:
        return np.float16
    elif data_type == 11:
        return np.float64
    elif data_type == 12:
        return np.uint32
    elif data_type == 13:
        return np.uint64
    else:
        return np.float32

def get_attr_data(node):
    attr = node.attribute[0]
    name, values  = onnx_attribute_to_dict(attr)

    if attr.name == name:
        return values
    elif attr.t.raw_data:
        values = np.frombuffer(attr.t.raw_data, dtype=onnx_datatype_to_npType(attr.t.data_type))
        return values
    else:
        assert (False), "Unsupported constant attribute type, node name = %s" % node.name

def Constant_to_initializer(onnxmodel):
    graph = onnxmodel.graph
    delete = []
    for i in range(len(graph.node)):
        if graph.node[i].op_type=="Constant":
            data_type = graph.node[i].attribute[0].t.data_type
            data_dims = graph.node[i].attribute[0].t.dims

            if graph.node[i].attribute[0].t.raw_data:
                data = np.frombuffer(graph.node[i].attribute[0].t.raw_data, dtype=onnx_datatype_to_npType(data_type))
            else:
                data = get_attr_data(graph.node[i])
            
            assert data_type != 0, "Unupported constant data type. Node: %s." % (graph.node[i].name)

            data = data.flatten()
            p_t = helper.make_tensor(graph.node[i].output[0], data_type, dims=data_dims, vals=data, raw=False)
            delete.append(graph.node[i])
            graph.initializer.append(p_t)
    for oldnode in delete:
        graph.node.remove(oldnode)

def check_names(onnx_model : onnx.ModelProto):
    def already_exist(new_name):
        already_exist = False
        for node in onnx_model.graph.node:
            if new_name == node.name:
                already_exist = True
                break
        return already_exist
    for i, node in enumerate(onnx_model.graph.node):
        if node.name == '':
            new_name = node.op_type + '_' + str(i)
            while already_exist(new_name):
                new_name = new_name + f"_{i}"
            onnx_model.graph.node[i].name = new_name

def rewrite_onnx_for_custom_layer(onnx_model):
    """
    Targeting node_rsqrt for qwen3-0.6B with Trojan Horse PyOp attributes
    """
    target_name = "node_rsqrt" # Target op
    target_node = None

    for i, node in enumerate(onnx_model.graph.node):
        if node.name == target_name:
            target_node = node
            break

    if target_node is None:
        print(f"找不到節點: {target_name}")
        return onnx_model

    print(f"找到目標節點: {target_node.name} (OpType: {target_node.op_type})")

    # 1. 改變節點名稱以匹配 cust_config.txt 中的 [name]
    new_name = target_node.name + "_pyop"
    target_node.name = new_name

    # 2. 核心技巧：【不要】改變 op_type，保留原來的標準算子 (如 Reciprocal)
    # 這樣 ONNX 的 Shape Inference 才能成功推導下游形狀
    # target_node.op_type = "PyOp"  <--- 絕對不要加這行

    # 3. 直接將底層 C++ Parser 需要的自定義屬性擴充 (extend) 到原節點上
    target_node.attribute.extend([
        helper.make_attribute("module", "my_custom_cpu_op"),
        helper.make_attribute("class_name", "MyReciprocal"),
        helper.make_attribute("input_types", [1]),  # 1 代表 TensorProto.FLOAT (float32)
        helper.make_attribute("output_types", [1]), # 1 代表 TensorProto.FLOAT (float32)
        helper.make_attribute("layer_name", new_name),
        helper.make_attribute("compute", "compute"),
        helper.make_attribute("scale_factor", "1.0")
    ])

    print(f"Rewriting successfully")
    return onnx_model

# def fix_tied_weights_bug(onnx_model):
#     """
#     修復 HuggingFace 導出 Tied Weights 時,lm_head.weight 遺失的 Bug
#     """
#     # 取得所有 initializer 的名稱
#     init_names = {init.name for init in onnx_model.graph.initializer}
    
#     # 檢查 lm_head.weight 是否真的變成了懸空輸入 (存在於節點輸入，但不在 initializer)
#     if "model.lm_head.weight" not in init_names:
#         print("Detected missing 'model.lm_head.weight' due to tied weights bug.")
        
#         # 尋找 Embedding 的權重名稱 (通常包含 embed_tokens.weight)
#         embed_name = None
#         for name in init_names:
#             if "embed_tokens.weight" in name:
#                 embed_name = name
#                 break
        
#         if embed_name:
#             print(f"Fixing tied weights: pointing 'model.lm_head.weight' to '{embed_name}'")
#             # 遍歷所有節點，把輸入是 lm_head.weight 的改成實際存在的 embed_tokens 權重
#             for node in onnx_model.graph.node:
#                 for i, inp in enumerate(node.input):
#                     if inp == "model.lm_head.weight":
#                         node.input[i] = embed_name
    
#     return onnx_model

def hf2onnx_pipline(args):

    if args.close_third_party_library_log==0:
        # 关闭 transformers 的 INFO/WARNING 日志
        import logging
        import warnings
        from transformers import logging as transformers_logging

        # 关闭 transformers 日志
        transformers_logging.set_verbosity_error()

        # 关闭 Python 的 warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # 关闭 PyTorch 的 TracerWarning
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        #-------------------

    if args.hf2onnx_method not in M2OREG.keys():
        print("Error : method invalid please check input , help: Supported Model2Onnx_Auto , Model2Onnx_Manual_NoCach , Model2Onnx_Manual_Cach")
        exit()
    
    if args.model_processor not in AutoModelProcessorDict.keys():
        support_name = ""
        for n in AutoModelProcessorDict.keys():
            support_name += " "
            support_name += str(n)
        print("Error : model_processor invalid please check input , help: Supported %s"%support_name)
        exit()
    
    HfModel2Onnx = M2OREG.get(args.hf2onnx_method)()
    HfModel2Onnx.set_auto_model_processer(model_type=args.model_processor)
    HfModel2Onnx.load_model(args.input)
    config = HfModel2Onnx.model.config
    HfModel2Onnx.get_export_cfg(args.input_shape)

    hf2onnx_dir = os.path.join(args.output,"hf_onnx")
    if not os.path.exists(hf2onnx_dir):
        os.mkdir(hf2onnx_dir)
    hf2onnx_path = os.path.join(hf2onnx_dir,'hf2onnx.onnx')
    HfModel2Onnx.export_onnx(hf2onnx_path)
    HfModel2Onnx.set_args_to_onnx2novaonnx_args(args)

    onnx_model = onnx.load(hf2onnx_path)


    # LLM 的后处理
    if args.model_processor == 'CausalLM':
        #onnx_model = onnx.load(hf2onnx_path)
        Constant_to_initializer(onnx_model)                 # 将constant node 中的权重放置到initializer
        check_names(onnx_model)
        onnx_model = simplify_hf_onnx(onnx_model,args,config)
        onnx_model=add_atten_in(onnx_model,HfModel2Onnx.atten_mask_shape)
        #save_onnx(onnx_model,os.path.join(args.output,'deploy_float.onnx'))
    else:
        #onnx_model = onnx.load(hf2onnx_path)
        #save_onnx(onnx_model,os.path.join(args.output,'deploy_float.onnx'))
        pass

    # for custom op
    onnx_model = rewrite_onnx_for_custom_layer(onnx_model)
    #onnx_model = fix_tied_weights_bug(onnx_model)
    save_onnx(onnx_model,os.path.join(args.output,'deploy_float.onnx'))

    return 




hf_method_name      = list(M2OREG.keys())
mode_processor_name = list(AutoModelProcessorDict.keys())

if __name__ == '__main__':
    pass

