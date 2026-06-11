import numpy as np

class RsqrtPyOp:
    def __init__(self, **kwargs):
        self.layer_name = kwargs.get('layer_name', 'default_layer')
        self.epsilon = 1e-5

    def reshape(self, in_shapes, out_shapes):
        input_shape = in_shapes[0]
        buf = out_shapes[0]
        
        try:
            # 嘗試直接在記憶體中將 C++ 的 buffer 縮放回真正的維度 (例如 3D)
            buf.resize(len(input_shape), refcheck=False)
            buf[:] = input_shape
        except Exception:
            # 如果 C++ 鎖定 buffer 長度為 4 不給縮放，我們必須「從左邊補 1」
            # 這樣廣播時尾巴才會對齊！(例如 [1, 512, 1] -> [1, 1, 512, 1])
            # 絕對不能從右邊補 1，否則會引發 512x512=262144 的廣播災難！
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
        out_data[0][:] = in_data[0][:]
        print("Input type: ", in_data[0].flatten()[:5])
        print("Input value: ", in_data[0].dtype)