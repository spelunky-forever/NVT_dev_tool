import numpy as np

class RsqrtPyOp:
    def __init__(self, **kwargs):
        self.layer_name = kwargs.get('layer_name', 'default_layer')
        self.epsilon = 1e-5

    def reshape(self, in_shapes, out_shapes):
        input_shape = in_shapes[0]
        buf = out_shapes[0]
        
        try:
            buf.resize(len(input_shape), refcheck=False)
            buf[:] = input_shape
        except Exception:
            pad_len = len(buf) - len(input_shape)
            for i in range(len(buf)):
                if i < pad_len:
                    buf[i] = 1
                else:
                    buf[i] = input_shape[i - pad_len]

    def compute(self, in_data, out_data):
        """
        :param in_data:  list of numpy array (dtype: float32)。包含實際的輸入特徵圖資料。
        :param out_data: list of numpy array (dtype: float32)。C++ 已配置好全 0 的記憶體。
        """
        x = in_data[0]
        result = 1.0 / np.sqrt(x + self.epsilon)

        out_data[0][:] = result