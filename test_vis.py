import numpy as np
import cv2
import tifffile as tiff
from dataset import Dataset
from easydict import EasyDict as Edict

# ================== CONFIG ==================
gt_path   = r"F:\Full-Dataset\FisheyeDepthDataset\omnithings\omnidepth_gt_640_raw\00001.tiff"
pred_path = r"F:\hyp_omni_h6\results\hyp_sync_1\invdepth_00001_romnistereo32_v3_e70.tiff"

dbname = "hyp_sync_1"
db_root = "./omnidata"

NEW_WIDTH  = 640
NEW_HEIGHT = 320
max_depth_view = 30.0
# ==========================================

# ========== INIT DATASET (để đọc đúng invdepth) ==========
opts = Edict()
opts.use_rgb = False
opts.num_invdepth = 192
opts.num_downsample = 1
opts.phi_deg = 45
opts.equirect_size = [160, 640]

data = Dataset(dbname, opts, db_root=db_root, train=False)

# ========== LOAD GT LOCAL ==========
gt = tiff.imread(gt_path).astype(np.float32)
gt[np.isnan(gt)] = 0
gt[np.isinf(gt)] = 0

gt = cv2.resize(gt, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

# ========== LOAD PRED (invdepth ERP) ==========
pred_invdepth = data.readInvdepth(pred_path)
pred_depth = 1.0 / pred_invdepth  # convert to depth thật

pred_depth[np.isnan(pred_depth)] = 0
pred_depth[np.isinf(pred_depth)] = 0

pred_depth = cv2.resize(pred_depth, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_NEAREST)

# ========== NORMALIZE FOR VIEW ==========
def make_vis(depth):
    depth_clip = np.clip(depth, 0, max_depth_view)
    norm = cv2.normalize(depth_clip, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return color

gt_color   = make_vis(gt)
pred_color = make_vis(pred_depth)

# ========== CLICK VIEW ==========
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= gt.shape[0] or x >= gt.shape[1]:
            return
        
        gt_val   = gt[y, x]
        pred_val = pred_depth[y, x]

        print(f"(x={x}, y={y}) | GT: {gt_val:.4f} | Pred: {pred_val:.4f}")

        show = np.hstack([gt_color, pred_color])
        cv2.circle(show, (x, y), 5, (0,255,0), -1)
        cv2.circle(show, (x+NEW_WIDTH, y), 5, (0,255,0), -1)

        cv2.putText(show, f"GT: {gt_val:.3f}", (x+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        cv2.putText(show, f"Pred: {pred_val:.3f}", (x+NEW_WIDTH+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        cv2.imshow("Compare", show)

# ========== SHOW ==========
print("Click để xem depth GT vs Pred")

vis = np.hstack([gt_color, pred_color])
cv2.namedWindow("Compare", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Compare", NEW_WIDTH*2, NEW_HEIGHT)
cv2.setMouseCallback("Compare", click_event)
cv2.imshow("Compare", vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
