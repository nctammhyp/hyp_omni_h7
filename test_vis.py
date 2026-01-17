import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize(img):
    img = img.astype(np.float32)
    minv = np.min(img)
    maxv = np.max(img)
    return (img - minv) / (maxv - minv + 1e-8)

def main(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("❌ Cannot load:", path)
        return

    print("Loaded:", path)
    print("dtype:", img.dtype)
    print("shape:", img.shape)
    print("min/max:", img.min(), img.max())

    # ===== CASE 1: Already RGB (colormapped) =====
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("Detected: Color image → show directly")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 4))
        plt.imshow(img_rgb)
        plt.title("Colormapped Prediction")
        plt.axis('off')
        plt.show()
        return

    # ===== CASE 2: Single channel =====
    img = img.astype(np.float32)
    minv = img.min()
    maxv = img.max()

    print("Depth min/max:", minv, maxv)

    img_norm = normalize(img)

    plt.figure(figsize=(12, 4))
    plt.imshow(img_norm, cmap='jet')
    plt.colorbar()
    plt.title("Prediction (normalized + jet)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vis.py path_to_prediction")
        exit()

    main(sys.argv[1])
