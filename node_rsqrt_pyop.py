import numpy as np

class RsqrtPyOp:
    def __init__(self, **kwargs):
        self.scale_factor = float(kwargs.get('scale_factor', '1.0'))
        self.layer_name = kwargs.get('layer_name', 'default_layer')
        
        self.epsilon = 1e-5

    def reshape(self, in_shapes, out_shapes):
        """
        :param in_shapes:  list of 1D numpy array (dtype: int32)。例如 in_shapes[0] 是 input_0 的 shape。
        :param out_shapes: list of 1D numpy array (dtype: int32)。長度已被 C++ 強制鎖定為 4。
        """
        input_shape = in_shapes[0]
        dims = len(input_shape)
        
        out_shapes[0][0] = input_shape[0] if dims > 0 else 1
        out_shapes[0][1] = input_shape[1] if dims > 1 else 1
        out_shapes[0][2] = input_shape[2] if dims > 2 else 1
        out_shapes[0][3] = input_shape[3] if dims > 3 else 1

    def compute(self, in_data, out_data):
        """
        :param in_data:  list of numpy array (dtype: float32)。包含實際的輸入特徵圖資料。
        :param out_data: list of numpy array (dtype: float32)。C++ 已配置好全 0 的記憶體。
        """
        x = in_data[0]
        result = self.scale_factor * (1.0 / np.sqrt(x + self.epsilon))

        out_data[0][:] = result