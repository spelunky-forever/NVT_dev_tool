import numpy as np

class MyReciprocal:
    def __init__(self):
        print(">> [PyOp] MyReciprocal CPU Operator 初始化成功！")

    def forward(self, inputs):
        x = inputs[0] 
        
        # 執行 Reciprocal 取倒數的操作 (1 / x)
        # 加入 where=x!=0 避免除以 0 導致程式崩潰
        out = np.divide(1.0, x, out=np.zeros_like(x), where=x!=0)
        
        print(f">> [PyOp] 處理了一個維度為 {x.shape} 的張量")
        
        # 必須將結果包裝成 list 或 tuple 回傳，C++ 才能正確解析
        return [out.astype(np.float32)]