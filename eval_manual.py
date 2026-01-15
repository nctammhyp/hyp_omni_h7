import os
import torch
import numpy as np
import cv2
import argparse
from easydict import EasyDict as Edict

# Import modules
from dataset import Dataset
from module.network import ROmniStereo
from utils.image import normalizeImage, rgb2gray, colorMap, imrescale, concat

def parse_args():
    parser = argparse.ArgumentParser(description='Manual Inference with 3 Images')
    parser.add_argument('--checkpoint_path', default = r"F:\hyp_omni_h6\checkpoints\romnistereo32_v1_e0.pth", help="Path to .pth file")
    
    # 3 ảnh đầu vào (Trước, Phải, Trái)
    parser.add_argument('--img1', default=r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\cam1\00124.png", help="Path to image Cam 1 (Front)")
    parser.add_argument('--img2', default=r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\cam2\00124.png", help="Path to image Cam 2 (Right)")
    parser.add_argument('--img3', default=r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\cam3\00124.png", help="Path to image Cam 3 (Left)")

    parser.add_argument('--output_path', default='manual_result.jpg', help="File kết quả đầu ra")
    
    # Cấu hình Model & Dataset (Cần để load Grid)
    parser.add_argument('--base_channel', type=int, default=32)
    parser.add_argument('--db_root', default=r'F:\Full-Dataset\FisheyeDepthDataset', type=str)
    parser.add_argument('--dbname', nargs='+', default=['omnithings'])
    
    # Mặc định
    parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640])
    parser.add_argument('--num_invdepth', type=int, default=192)
    parser.add_argument('--phi_deg', type=float, default=45.0)
    parser.add_argument('--num_downsample', type=int, default=1)
    
    return parser.parse_args()

def load_image_manual(path, target_size=(800, 768), mask=None):
    """Đọc, resize, gray và normalize 1 ảnh"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
        
    # Đọc bằng CV2
    img = cv2.imread(path)
    if img is None: raise ValueError(f"Cannot read image: {path}")
    
    # Resize về kích thước mạng yêu cầu
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Giữ bản gốc để visualize
    raw_img = img.copy()
    
    # Chuyển sang Gray
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32)
    else:
        img_gray = img.astype(np.float32)
    
    # Normalize (Thủ công để đảm bảo an toàn)
    img_gray = img_gray / 255.0
    
    # Apply mask (nếu có)
    if mask is not None:
        if mask.ndim == 3: mask = mask[:, :, 0]
        # Resize mask về đúng kích thước ảnh nếu cần
        if mask.shape != img_gray.shape:
             mask = cv2.resize(mask.astype(np.uint8), (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        img_gray[mask.astype(bool)] = 0.0
        
    # Thêm dimension: [H, W] -> [1, 1, H, W] (Batch, Channel, H, W)
    tensor_img = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float()
    
    return tensor_img, raw_img

def main():
    args = parse_args()
    
    # 1. Init Dataset (Chỉ để lấy Grids và Masks)
    opts = Edict()
    opts.phi_deg = args.phi_deg
    opts.num_invdepth = args.num_invdepth
    opts.equirect_size = args.equirect_size
    opts.num_downsample = args.num_downsample
    
    print("Initializing geometry (grids)...")
    ds_dummy = Dataset(args.dbname[0], opts, load_lut=True, train=False, db_root=args.db_root)
    
    # --- FIX LỖI Ở ĐÂY: KHÔNG DÙNG unsqueeze(0) ---
    grids = [torch.tensor(g, requires_grad=False).cuda() for g in ds_dummy.grids]
    masks = [cam.invalid_mask for cam in ds_dummy.ocams]

    # 2. Load Images
    print("Loading images...")
    target_size = (800, 768) 
    
    t1, r1 = load_image_manual(args.img1, target_size, masks[0])
    t2, r2 = load_image_manual(args.img2, target_size, masks[1])
    t3, r3 = load_image_manual(args.img3, target_size, masks[2])
    
    imgs_input = [t1.cuda(), t2.cuda(), t3.cuda()]

    # 3. Load Model
    print("Loading model...")
    net_opts = Edict()
    net_opts.base_channel = args.base_channel
    net_opts.num_invdepth = args.num_invdepth
    net_opts.use_rgb = False
    net_opts.encoder_downsample_twice = False
    net_opts.num_downsample = args.num_downsample
    net_opts.corr_levels = 4
    net_opts.corr_radius = 4
    net_opts.mixed_precision = False
    
    net = ROmniStereo(net_opts).cuda()
    
    # Load weights
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    state_dict = checkpoint['net_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.eval()
    
    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        pred_idx = net(imgs_input, grids, iters=12, test_mode=True)
        
    # 5. Visualization
    pred_idx = pred_idx.cpu().numpy()[0, 0] # [H, W]
    pred_invdepth = ds_dummy.indexToInvdepth(pred_idx)
    pred_color = colorMap('oliver', pred_invdepth, ds_dummy.min_invdepth, ds_dummy.max_invdepth)
    
    # Ghép ảnh input
    input_vis = np.concatenate([r1, r2, r3], axis=1)
    
    scale_factor = input_vis.shape[1] / pred_color.shape[1]
    pred_color_resized = cv2.resize(pred_color, (input_vis.shape[1], int(pred_color.shape[0] * scale_factor)))
    
    final_vis = np.concatenate([input_vis, pred_color_resized], axis=0)
    
    cv2.imwrite(args.output_path, final_vis)
    print(f"Saved result to {args.output_path}")

if __name__ == '__main__':
    main()