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
    # Load image
    if path.endswith(".npy"):
        img = np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("❌ Cannot load file:", path)
        return

    print("Loaded:", path)
    print("dtype:", img.dtype)
    print("shape:", img.shape)
    print("min/max:", img.min(), img.max())

    # Normalize for visualization
    img_norm = normalize(img)

    # Show
    plt.figure(figsize=(12, 4))
    plt.imshow(img_norm, cmap='jet')
    plt.colorbar()
    plt.title(path)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_pred.py your_prediction.tiff")
        sys.exit(0)

    main(sys.argv[1])
