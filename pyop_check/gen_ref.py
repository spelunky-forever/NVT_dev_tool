import numpy as np

def generate_input_bin(output_path="ref_input_X2.bin"):
    # 設定與模型一致的輸入 Shape
    input_shape = (1, 512, 1)
    
    # 生成隨機輸入資料
    # 這裡我們生成 1.0 ~ 5.0 之間的浮點數 (float32)
    np.random.seed(42) # 固定 seed，確保每次生出來的 bin 都一樣
    x_data = np.random.uniform(low=1.0, high=5.0, size=input_shape).astype(np.float32)

    # 直接儲存為純二進位格式 (.bin)
    x_data.tofile(output_path)

    print(f"✅ 成功生成純輸入的 bin 檔: {output_path}")
    print(f"  - Data Shape: {input_shape}")
    print(f"  - Data Type: float32")
    print(f"  - [預覽] 前三個值: {x_data.flatten()[:3]}")

if __name__ == "__main__":
    generate_input_bin()