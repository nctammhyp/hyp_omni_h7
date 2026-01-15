import os
import glob
import numpy as np
import tifffile  # Thư viện dùng trong project của bạn
import cv2
from tqdm import tqdm

# ================= CẤU HÌNH =================
# Đường dẫn đến folder GT hiện tại (folder gây lỗi)
INPUT_DIR = r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\omnidepth_gt_640_backup"

# Đường dẫn đến folder mới (sẽ chứa ảnh đã sửa)
OUTPUT_DIR = r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\omnidepth_gt_640"
# ============================================

def convert_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục output: {OUTPUT_DIR}")

    # Lấy danh sách file .tiff hoặc .tif
    files = glob.glob(os.path.join(INPUT_DIR, "*.tif*"))
    
    if len(files) == 0:
        print("Không tìm thấy file .tiff nào trong thư mục input!")
        return

    print(f"Tìm thấy {len(files)} ảnh. Bắt đầu chuyển đổi...")

    for file_path in tqdm(files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_DIR, filename)

        try:
            # Dùng tifffile để đọc (giữ nguyên chuẩn float32 nếu có)
            img = tifffile.imread(file_path)

            # --- XỬ LÝ CHUYỂN VỀ 2D ---
            # Nếu ảnh có 3 chiều (H, W, 3) hoặc (H, W, 4) -> Lấy kênh đầu tiên
            if img.ndim == 3:
                img = img[:, :, 0]
            
            # Đảm bảo array là contiguous (liên tục trong bộ nhớ) để tránh lỗi lạ
            img = np.ascontiguousarray(img)

            # Lưu lại file mới
            tifffile.imwrite(out_path, img)

        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {e}")

    print("\nHoàn tất! Hãy đổi tên folder cũ và thay thế bằng folder mới.")

if __name__ == "__main__":
    convert_dataset()