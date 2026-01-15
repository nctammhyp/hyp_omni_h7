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
from utils.image import colorMap, imrescale, concat

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on DataLoader')
    parser.add_argument('--name', default='romnistereo32_v1', help="Tên experiment")
    parser.add_argument('--checkpoint_path', required=True, help="Đường dẫn file .pth (ví dụ: checkpoints/.../e29.pth)")
    parser.add_argument('--output_dir', default='./outputs/dataloader_vis', help="Nơi lưu ảnh kết quả")
    
    # Cấu hình giống lúc train
    parser.add_argument('--base_channel', type=int, default=32)
    parser.add_argument('--num_invdepth', type=int, default=192)
    parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640])
    parser.add_argument('--db_root', default=r'F:\Full-Dataset\FisheyeDepthDataset', type=str)
    parser.add_argument('--dbname', nargs='+', default=['omnithings'])
    
    # Các tham số mặc định khác
    parser.add_argument('--phi_deg', type=float, default=45.0)
    parser.add_argument('--num_downsample', type=int, default=1)
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--encoder_downsample_twice', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--fix_bn', action='store_true')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Options
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

    # 2. Load Dataset (Mode Train=False để lấy tập test)
    print(f"Loading dataset {args.dbname}...")
    dataset = Dataset(args.dbname[0], opts.data_opts, load_lut=True, train=False, db_root=args.db_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    net = ROmniStereo(opts.net_opts).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    state_dict = checkpoint['net_state_dict']
    
    # Xử lý key 'module.' nếu train bằng DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    
    net.eval()

    # 4. Prepare Grids
    grids = [torch.tensor(g, requires_grad=False).cuda().unsqueeze(0) for g in dataset.grids]

    # 5. Output Directory
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    # 6. Inference Loop
    with torch.no_grad():
        for i, data_blob in enumerate(tqdm(dataloader)):
            imgs, gt, valid, raw_imgs = data_blob
            
            # Move images to GPU
            imgs = [img.float().cuda() for img in imgs]
            
            # Forward pass
            # grids cần repeat theo batch size (ở đây batch=1 nên ok, nếu batch>1 cần repeat)
            current_grids = [g.repeat(imgs[0].shape[0], 1, 1, 1, 1).squeeze(0) for g in grids]
            
            # Predict
            pred_idx = net(imgs, current_grids, iters=12, test_mode=True)
            
            # --- Visualization ---
            pred_idx = pred_idx.cpu().numpy()[0, 0] # Lấy kết quả đầu tiên, kênh đầu tiên
            gt = gt.cpu().numpy()[0]
            
            # Convert Index to InvDepth
            pred_invdepth = dataset.indexToInvdepth(pred_idx)
            
            # Convert raw images to numpy for visualization
            vis_raw_imgs = [toNumpy(raw[0]) for raw in raw_imgs]
            
            # Sử dụng hàm makeVisImage có sẵn trong dataset
            # Nó sẽ ghép (Input | Pred | GT Error)
            vis_img = dataset.makeVisImage(vis_raw_imgs, pred_invdepth, gt=gt)
            
            # Lưu ảnh
            frame_id = dataset.test_idx[i]
            save_path = os.path.join(save_dir, f"result_{frame_id:05d}.jpg")
            
            # Chuyển từ RGB (matplotlib/dataset) sang BGR (opencv) để lưu đúng màu
            if vis_img.shape[2] == 3:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                
            cv2.imwrite(save_path, vis_img)

    print("Done!")

if __name__ == '__main__':
    main()