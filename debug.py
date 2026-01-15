import numpy as np
import cv2
import tifffile
import os

# Thay đường dẫn này bằng đường dẫn tới 1 file GT trong dataset MỚI của bạn
FILE_PATH = r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\omnidepth_gt_640\00268.tiff"
# (Hoặc file .tif bất kỳ trong thư mục đó)

def inspect_tiff(path):
    if not os.path.exists(path):
        print("File không tồn tại!")
        return

    try:
        # Thử đọc bằng tifffile
        img = tifffile.imread(path)
        print(f"--- Kiểm tra file: {os.path.basename(path)} ---")
        print(f"Shape: {img.shape}")
        print(f"Dtype: {img.dtype}")
        
        # Nếu 3 kênh, lấy kênh đầu
        if img.ndim == 3:
            img = img[:, :, 0]
            print("-> Đã lấy kênh đầu tiên để kiểm tra.")

        print(f"Min value: {np.nanmin(img)}")
        print(f"Max value: {np.nanmax(img)}")
        print(f"Có NaN không?: {np.isnan(img).any()}")
        print(f"Có Inf không?: {np.isinf(img).any()}")
        
        # Kiểm tra giá trị âm (thường là lỗi trong synthetic data)
        print(f"Có giá trị âm không?: {(img < 0).any()}")

    except Exception as e:
        print(f"Lỗi đọc file: {e}")

if __name__ == "__main__":
    inspect_tiff(FILE_PATH)