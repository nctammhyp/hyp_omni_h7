import os
import torch
import numpy as np
import cv2
import argparse
from easydict import EasyDict as Edict
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các module của project
from dataset import Dataset
from module.network import ROmniStereo
from utils.common import toNumpy

# --- CẤU HÌNH VISUALIZE ---
MAX_DEPTH_VIEW = 30.0  # Clip 30m để hiển thị màu Inferno

def parse_args():
    parser = argparse.ArgumentParser(description='Inference, Calculate RMSE & Visualize')
    parser.add_argument('--checkpoint_path', required=True, help="Đường dẫn file .pth")
    parser.add_argument('--output_dir', default='./outputs/eval_rmse', help="Nơi lưu ảnh kết quả")
    
    # Cấu hình Dataset/Model (Phải khớp config train)
    parser.add_argument('--db_root', default=r'F:\Full-Dataset\FisheyeDepthDataset', type=str)
    parser.add_argument('--dbname', nargs='+', default=['omnithings'])
    parser.add_argument('--num_invdepth', type=int, default=192)
    parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640])
    parser.add_argument('--base_channel', type=int, default=32)
    
    # Default params
    parser.add_argument('--phi_deg', type=float, default=45.0)
    parser.add_argument('--num_downsample', type=int, default=1)
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--encoder_downsample_twice', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    
    return parser.parse_args()

def convert_to_vis_format_C(depth_meters):
    """
    Convert Depth (m) -> Inferno Color Map (Clip 30m)
    """
    # Xử lý NaN/Inf
    depth_meters = np.nan_to_num(depth_meters, nan=0.0, posinf=MAX_DEPTH_VIEW, neginf=0.0)
    # Clip 0-30m
    out_clipped = np.clip(depth_meters, 0, MAX_DEPTH_VIEW)
    # Normalize 0-255
    out_norm = cv2.normalize(out_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply Inferno
    out_color = cv2.applyColorMap(out_norm, cv2.COLORMAP_INFERNO)
    return out_color

def calculate_rmse_meters(pred_depth_m, gt_depth_m, valid_mask):
    """
    Tính RMSE theo đơn vị Mét (loại bỏ các điểm không valid)
    """
    if valid_mask.sum() == 0:
        return 0.0
    
    diff = pred_depth_m[valid_mask] - gt_depth_m[valid_mask]
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    return rmse

def main():
    args = parse_args()
    
    # 1. Setup Configs
    opts = Edict()
    opts.data_opts = Edict()
    opts.data_opts.phi_deg = args.phi_deg
    opts.data_opts.num_invdepth = args.num_invdepth
    opts.data_opts.equirect_size = args.equirect_size
    opts.data_opts.num_downsample = args.num_downsample
    opts.data_opts.use_rgb = args.use_rgb

    opts.net_opts = Edict()
    opts.net_opts.base_channel = args.base_channel
    opts.net_opts.num_invdepth = args.num_invdepth
    opts.net_opts.use_rgb = args.use_rgb
    opts.net_opts.encoder_downsample_twice = args.encoder_downsample_twice
    opts.net_opts.num_downsample = args.num_downsample
    opts.net_opts.corr_levels = args.corr_levels
    opts.net_opts.corr_radius = args.corr_radius
    opts.net_opts.mixed_precision = args.mixed_precision

    # 2. Load Dataset (Tập Test)
    print(f"Loading dataset {args.dbname}...")
    dataset = Dataset(args.dbname[0], opts.data_opts, load_lut=True, train=False, db_root=args.db_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    net = ROmniStereo(opts.net_opts).cuda()
    
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    state_dict = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.eval()

    grids = [torch.tensor(g, requires_grad=False).cuda().unsqueeze(0) for g in dataset.grids]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Start Processing & Calculating RMSE...")
    
    total_rmse_idx = 0.0
    total_rmse_meter = 0.0
    count = 0

    with torch.no_grad():
        for i, data_blob in enumerate(tqdm(dataloader)):
            imgs, gt, valid, raw_imgs = data_blob
            
            # --- A. INFERENCE ---
            imgs = [img.float().cuda() for img in imgs]
            current_grids = [g.repeat(imgs[0].shape[0], 1, 1, 1, 1).squeeze(0) for g in grids]
            
            # Prediction (Index)
            pred_idx = net(imgs, current_grids, iters=12, test_mode=True)
            
            # Lấy data numpy từ tensor (Batch size = 1)
            pred_idx_np = pred_idx.cpu().numpy()[0, 0] # (H, W)
            gt_idx_np = gt.cpu().numpy()[0]            # (H, W)
            valid_np = valid.cpu().numpy()[0].astype(bool)

            # --- B. CALCULATE ERROR (Sample-wise) ---
            
            # 1. Tính lỗi dựa trên Index (Chuẩn của database.py)
            # Hàm evalError trả về: e1, e3, e5, mae, rms
            _, _, _, _, rmse_idx = dataset.evalError(pred_idx_np, gt_idx_np, valid_np)
            
            # 2. Tính lỗi dựa trên Depth Mét (Để visualize cho trực quan)
            pred_inv = dataset.indexToInvdepth(pred_idx_np)
            gt_inv = dataset.indexToInvdepth(gt_idx_np)
            
            with np.errstate(divide='ignore'):
                pred_depth_m = 1.0 / pred_inv
                gt_depth_m = 1.0 / gt_inv
            
            # Lọc mask hợp lệ (gt >= 0 và gt < num_invdepth)
            valid_mask_meter = valid_np & (gt_idx_np >= 0) & (gt_idx_np < args.num_invdepth)
            rmse_meter = calculate_rmse_meters(pred_depth_m, gt_depth_m, valid_mask_meter)

            # Cộng dồn để tính trung bình sau cùng
            total_rmse_idx += rmse_idx
            total_rmse_meter += rmse_meter
            count += 1
            
            # --- C. VISUALIZE (Format C) ---
            
            # Convert Pred & GT sang ảnh màu Inferno
            vis_pred = convert_to_vis_format_C(pred_depth_m)
            # GT cần xóa những vùng không valid đi cho đẹp
            gt_depth_m[~valid_mask_meter] = 0
            vis_gt = convert_to_vis_format_C(gt_depth_m)

            # Chuẩn bị ảnh Input (ghép 3 ảnh Fisheye)
            fisheye_list = []
            for img in raw_imgs:
                img_np = toNumpy(img[0])
                if img_np.max() <= 1.0: img_np = (img_np * 255).astype(np.uint8)
                else: img_np = img_np.astype(np.uint8)
                
                if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                elif img_np.shape[0] == 3: img_np = img_np.transpose(1, 2, 0)
                
                # Resize cho đẹp
                h, w = img_np.shape[:2]
                fisheye_list.append(cv2.resize(img_np, (int(w*256/h), 256)))
            
            input_row = np.hstack(fisheye_list)

            # Resize Pred/GT theo chiều rộng Input
            target_w = input_row.shape[1]
            target_h = int(vis_pred.shape[0] * (target_w / vis_pred.shape[1]))
            vis_pred = cv2.resize(vis_pred, (target_w, target_h))
            vis_gt = cv2.resize(vis_gt, (target_w, target_h))

            # --- D. VIẾT THÔNG SỐ LÊN ẢNH ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Ghi lên Prediction
            cv2.putText(vis_pred, f"Prediction | RMSE (Index): {rmse_idx:.2f} | RMSE (Meter): {rmse_meter:.2f}m", 
                        (20, 40), font, 0.8, (0, 255, 0), 2)
            
            # Ghi lên GT
            cv2.putText(vis_gt, f"Ground Truth", (20, 40), font, 0.8, (0, 255, 0), 2)

            # Ghép và lưu
            final_img = np.vstack([input_row, vis_pred, vis_gt])
            frame_id = dataset.test_idx[i]
            
            # Lưu ảnh
            cv2.imwrite(os.path.join(args.output_dir, f"eval_{frame_id:05d}_rmse_{rmse_idx:.2f}.jpg"), final_img)

    # In kết quả trung bình toàn bộ tập test
    print(f"\n=== Evaluation Complete ===")
    print(f"Samples processed: {count}")
    print(f"Average RMSE (Index): {total_rmse_idx / count:.4f}")
    print(f"Average RMSE (Meters): {total_rmse_meter / count:.4f} m")

if __name__ == '__main__':
    main()