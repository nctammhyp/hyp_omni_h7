# # utils.image
# # 
# # Author: Changhee Won (changhee.1.won@gmail.com)
# #
# #
# import sys
# import os
# import os.path as osp
# import traceback
# import matplotlib
# import torch
# import torch.nn.functional as F
# import numpy as np
# import tifffile
# import skimage.io
# import skimage.transform
# from utils.common import *
# from utils.log import *

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# ## visualize =================================

# def colorMap(colormap_name: str, arr: np.ndarray,
#              min_v=None, max_v=None, alpha=None) -> np.ndarray:
#     arr = toNumpy(arr).astype(np.float64).squeeze()
#     if colormap_name == 'oliver': return colorMapOliver(arr, min_v, max_v)
#     cmap = matplotlib.cm.get_cmap(colormap_name)
#     if max_v is None: max_v = np.max(arr)
#     if min_v is None: min_v = np.min(arr)
#     arr[arr > max_v] = max_v
#     arr[arr < min_v] = min_v
#     arr = (arr - min_v) / (max_v - min_v)
#     if alpha is None:
#         out = cmap(arr)
#         out = out[:, :, 0:3]
#     else:
#         out = cmap(arr, alpha=alpha)
#     return np.round(255 * out).astype(np.uint8)

# #
# # code adapted from Oliver Woodford's sc.m
# _CMAP_OLIVER = np.array(
#     [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174], [0,1,0,114],
#      [0,1,1,185], [1,1,0,114], [1,1,1,0]]).astype(np.float64)
# #
# def colorMapOliver(arr: np.ndarray, min_v=None, max_v=None) -> np.ndarray:
#     arr = toNumpy(arr).astype(np.float64).squeeze()
#     height, width = arr.shape
#     arr = arr.reshape([1, -1])
#     if max_v is None: max_v = np.max(arr)
#     if min_v is None: min_v = np.min(arr)
#     arr[arr < min_v] = min_v
#     arr[arr > max_v] = max_v
#     arr = (arr - min_v) / (max_v - min_v)
#     bins = _CMAP_OLIVER[:-1, 3]
#     cbins = np.cumsum(bins)
#     bins = bins / cbins[-1]
#     cbins = cbins[:-1] / cbins[-1]
#     ind = np.sum(
#         np.tile(arr, [6, 1]) > \
#         np.tile(np.reshape(cbins,[-1, 1]), [1, arr.size]), axis=0)
#     ind[ind > 6] = 6
#     bins = 1 / bins
#     cbins = np.array([0.0] + cbins.tolist())
#     arr = (arr - cbins[ind]) * bins[ind]
#     arr = _CMAP_OLIVER[ind, :3] * np.tile(np.reshape(1 - arr,[-1, 1]),[1,3]) + \
#         _CMAP_OLIVER[ind+1, :3] * np.tile(np.reshape(arr,[-1, 1]),[1,3])
#     arr[arr < 0] = 0
#     arr[arr > 1] = 1
#     out = np.reshape(arr, [height, width, 3])
#     out = np.round(255 * out).astype(np.uint8)
#     return out

# ## image transform =================================

# def rgb2gray(I: np.ndarray, channel_wise_mean=True) -> np.ndarray:
#     I = toNumpy(I)
#     dtype = I.dtype
#     I = I.astype(np.float64)
#     if channel_wise_mean:
#         return np.mean(I, axis=2).squeeze().astype(dtype)
#     else:
#         return np.dot(I[...,:3], [0.299, 0.587, 0.114]).astype(dtype)

# def imrescale(image: np.ndarray, scale: float) -> np.ndarray:
#     image = toNumpy(image)
#     dtype = image.dtype
#     multi_channel = True if len(image.shape) == 3 else False
#     out = skimage.transform.rescale(image, scale, 
#         channel_axis=-1, preserve_range=True)
#     return out.astype(dtype)

# imresize = skimage.transform.resize

# def interp2D(I, grid):
#     istensor = type(I) == torch.Tensor
#     I = torch.tensor(I).float().squeeze().unsqueeze(0) # make 1 x C x H x W
#     grid = torch.Tensor(grid).squeeze().unsqueeze(0) # make 1 x npts x 2
#     if len(I.shape) < 4 : # if 1D channel image
#         I = I.unsqueeze(0)
#     out = F.grid_sample(I, grid, mode='bilinear', align_corners=True).squeeze()
#     if not istensor: out = out.numpy()
#     return out

# def pixelToGrid(pts, target_resolution: (int, int), 
#                 source_resolution: (int, int)):
#     h, w = target_resolution
#     height, width = source_resolution
#     xs = (pts[0,:]) / (width - 1) * 2 - 1
#     ys = (pts[1,:]) / (height - 1) * 2 - 1
#     xs = xs.reshape((h, w, 1))
#     ys = ys.reshape((h, w, 1))
#     return concat((xs, ys), 2)

# # def normalizeImage(image: np.ndarray, mask=None,
# #                    channel_wise_mean=True) -> np.ndarray:
# #     image = toNumpy(image)
# #     def __normalizeImage1D(image, mask):
# #         image = image.squeeze().astype(np.float32)
# #         if mask is not None: image[mask] = np.nan
# #         # normalize intensities
# #         # image = (image - np.nanmean(image.flatten())) / \
# #         #     np.nanstd(image.flatten())
# #         std_val = np.nanstd(image.flatten())
# #         if std_val < 1e-7:
# #             std_val = 1e-7 # Tránh chia cho 0
        
# #         image = (image - np.nanmean(image.flatten())) / std_val

# #         if mask is not None: image[mask] = 0
# #         return image
# #     if len(image.shape) == 3 and image.shape[2] == 3:
# #         if channel_wise_mean:
# #             return np.concatenate(
# #                 [__normalizeImage1D(image[:,:,i], mask)[..., np.newaxis] 
# #                     for i in range(3)], axis=2)
# #         else:
# #             image = image.squeeze().astype(np.float32)
# #             mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
# #             if mask is not None: image[mask] = np.nan
# #             # normalize intensities
# #             image = (image - np.nanmean(image.flatten())) / \
# #                 np.nanstd(image.flatten())
# #             if mask is not None: image[mask] = 0
# #             return image
# #     else:
# #         return __normalizeImage1D(image, mask)

# def normalizeImage(image: np.ndarray, mask=None,
#                    channel_wise_mean=True) -> np.ndarray:
#     image = toNumpy(image)
    
#     def __normalizeImage1D(image, mask):
#         image = image.squeeze().astype(np.float32)
#         if mask is not None: image[mask] = np.nan
        
#         # --- FIX: TRÁNH CHIA CHO 0 ---
#         mean_val = np.nanmean(image.flatten())
#         std_val = np.nanstd(image.flatten())
        
#         # Nếu std quá nhỏ (ảnh đồng màu hoặc đen), gán giá trị nhỏ để tránh lỗi
#         if std_val < 1e-6:
#             std_val = 1e-6
            
#         image = (image - mean_val) / std_val
#         # -----------------------------
        
#         if mask is not None: image[mask] = 0
#         return image

#     if len(image.shape) == 3 and image.shape[2] == 3:
#         if channel_wise_mean:
#             return np.concatenate(
#                 [__normalizeImage1D(image[:,:,i], mask)[..., np.newaxis] 
#                     for i in range(3)], axis=2)
#         else:
#             image = image.squeeze().astype(np.float32)
#             mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
#             if mask is not None: image[mask] = np.nan
            
#             # --- FIX CHO TRƯỜNG HỢP NÀY ---
#             mean_val = np.nanmean(image.flatten())
#             std_val = np.nanstd(image.flatten())
#             if std_val < 1e-6:
#                 std_val = 1e-6
#             image = (image - mean_val) / std_val
#             # ------------------------------
            
#             if mask is not None: image[mask] = 0
#             return image
#     else:
#         return __normalizeImage1D(image, mask)


# ## image file I/O =================================

# # def writeImageFloat(image: np.ndarray, tiff_path: str, thumbnail = None):
# #     image = toNumpy(image)
# #     with tifffile.TiffWriter(tiff_path) as tiff:
# #         if thumbnail is not None:
# #             if not thumbnail.dtype == np.uint8:
# #                 thumbnail = thumbnail.astype(np.uint8)
# #             tiff.save(thumbnail, photometric='RGB', planarconfig='CONTIG',
# #                 bitspersample=8)
# #         if not image.dtype == np.float32:
# #             image = image.astype(np.float32)
# #         tiff.save(image, photometric='MINISBLACK', planarconfig='CONTIG',
# #                 bitspersample=32, compress=9)

# def writeImageFloat(image: np.ndarray, tiff_path: str, thumbnail=None):
#     image = toNumpy(image)

#     with tifffile.TiffWriter(tiff_path) as tiff:
#         if thumbnail is not None:
#             if thumbnail.dtype != np.uint8:
#                 thumbnail = thumbnail.astype(np.uint8)
#             tiff.write(
#                 thumbnail,
#                 photometric='rgb',
#                 planarconfig='contig',
#                 bitspersample=8
#             )

#         if image.dtype != np.float32:
#             image = image.astype(np.float32)
#         tiff.write(
#             image,
#             photometric='minisblack',
#             planarconfig='contig',
#             bitspersample=32,
#             compression='zlib'  # <-- safe, supported
#         )



# def readImageFloat(tiff_path: str, return_thumbnail = False,
#                    read_or_die = True):
#     try:
#         multi_image = skimage.io.ImageCollection(tiff_path)
#         num_read_images = len(multi_image)
#         if num_read_images == 0:
#             raise Exception('No images found.')
#         elif num_read_images == 1:
#             return multi_image[0].squeeze(), None
#         elif num_read_images == 2: # returns float, thumnail
#             if multi_image[0].dtype == np.uint8:
#                 if not return_thumbnail: return multi_image[1].squeeze()
#                 else: return multi_image[1].squeeze(), multi_image[0].squeeze()
#             else:
#                 if not return_thumbnail: return multi_image[0].squeeze()
#                 else: return multi_image[0].squeeze(), multi_image[1].squeeze()
#         else: # returns list of images
#             return [im.squeeze() for im in multi_image]       
#     except Exception as e:
#         LOG_ERROR('Failed to read image float: "%s"' %(e))
#         if read_or_die:
#             traceback.print_tb(e.__traceback__)
#             sys.exit()
#         return None, None

# def writeImage(image: np.ndarray, path: str):
#     image = toNumpy(image)
#     skimage.io.imsave(path, image)

# import imageio.v3 as iio
# import traceback
# import sys

# def readImage(path: str, read_or_die=True):
#     """
#     Đọc ảnh từ đường dẫn. 
#     Nếu fail, in lỗi và thoát (mặc định read_or_die=True)
#     """
#     try:
#         return iio.imread(path)
#     except Exception as e:
#         print(f'[ERROR] Failed to read image: "{path}" -> {e}')
#         if read_or_die:
#             traceback.print_tb(e.__traceback__)
#             sys.exit(1)
#         return None


# utils.image
# 
# Fix: Safe Normalization & OpenCV Reading
#
import sys
import os
import os.path as osp
import traceback
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import skimage.io
import skimage.transform
from utils.common import *
from utils.log import *
import cv2  # Dùng OpenCV cho ổn định

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## visualize =================================

def colorMap(colormap_name: str, arr: np.ndarray,
             min_v=None, max_v=None, alpha=None) -> np.ndarray:
    arr = toNumpy(arr).astype(np.float64).squeeze()
    if colormap_name == 'oliver': return colorMapOliver(arr, min_v, max_v)
    cmap = matplotlib.cm.get_cmap(colormap_name)
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    # Tránh chia cho 0 trong visualize
    denom = max_v - min_v
    if denom == 0: denom = 1e-6
    
    arr[arr > max_v] = max_v
    arr[arr < min_v] = min_v
    arr = (arr - min_v) / denom
    
    if alpha is None:
        out = cmap(arr)
        out = out[:, :, 0:3]
    else:
        out = cmap(arr, alpha=alpha)
    return np.round(255 * out).astype(np.uint8)

# code adapted from Oliver Woodford's sc.m
_CMAP_OLIVER = np.array(
    [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174], [0,1,0,114],
     [0,1,1,185], [1,1,0,114], [1,1,1,0]]).astype(np.float64)

def colorMapOliver(arr: np.ndarray, min_v=None, max_v=None) -> np.ndarray:
    arr = toNumpy(arr).astype(np.float64).squeeze()
    height, width = arr.shape
    arr = arr.reshape([1, -1])
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    
    denom = max_v - min_v
    if denom == 0: denom = 1e-6

    arr[arr < min_v] = min_v
    arr[arr > max_v] = max_v
    arr = (arr - min_v) / denom
    
    bins = _CMAP_OLIVER[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = np.sum(
        np.tile(arr, [6, 1]) > \
        np.tile(np.reshape(cbins,[-1, 1]), [1, arr.size]), axis=0)
    ind[ind > 6] = 6
    bins = 1 / bins
    cbins = np.array([0.0] + cbins.tolist())
    arr = (arr - cbins[ind]) * bins[ind]
    arr = _CMAP_OLIVER[ind, :3] * np.tile(np.reshape(1 - arr,[-1, 1]),[1,3]) + \
        _CMAP_OLIVER[ind+1, :3] * np.tile(np.reshape(arr,[-1, 1]),[1,3])
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    out = np.reshape(arr, [height, width, 3])
    out = np.round(255 * out).astype(np.uint8)
    return out

## image transform =================================

def rgb2gray(I: np.ndarray, channel_wise_mean=True) -> np.ndarray:
    I = toNumpy(I)
    dtype = I.dtype
    I = I.astype(np.float64)
    if channel_wise_mean:
        return np.mean(I, axis=2).squeeze().astype(dtype)
    else:
        return np.dot(I[...,:3], [0.299, 0.587, 0.114]).astype(dtype)

def imrescale(image: np.ndarray, scale: float) -> np.ndarray:
    image = toNumpy(image)
    dtype = image.dtype
    out = skimage.transform.rescale(image, scale, 
        channel_axis=-1, preserve_range=True)
    return out.astype(dtype)

imresize = skimage.transform.resize

def interp2D(I, grid):
    istensor = type(I) == torch.Tensor
    I = torch.tensor(I).float().squeeze().unsqueeze(0) 
    grid = torch.Tensor(grid).squeeze().unsqueeze(0)
    if len(I.shape) < 4 : 
        I = I.unsqueeze(0)
    out = F.grid_sample(I, grid, mode='bilinear', align_corners=True).squeeze()
    if not istensor: out = out.numpy()
    return out

def pixelToGrid(pts, target_resolution: (int, int), 
                source_resolution: (int, int)):
    h, w = target_resolution
    height, width = source_resolution
    xs = (pts[0,:]) / (width - 1) * 2 - 1
    ys = (pts[1,:]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return concat((xs, ys), 2)

# =========================================================================
# HÀM QUAN TRỌNG NHẤT ĐÃ ĐƯỢC SỬA LẠI ĐỂ TRÁNH NAN
# =========================================================================
def normalizeImage(image: np.ndarray, mask=None, channel_wise_mean=True) -> np.ndarray:
    image = toNumpy(image).astype(np.float32)

    # 1. Scale về [0, 1]
    image = image / 255.0

    # 2. Xử lý Mask
    if mask is not None:
        # --- FIX QUAN TRỌNG: Ép Mask về 2D ---
        # Nếu mask có 3 chiều (H, W, 3), chỉ lấy kênh đầu tiên
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        # Đảm bảo mask là kiểu boolean chuẩn
        mask = mask.astype(bool)
        # -------------------------------------

        # Gán 0 cho vùng không hợp lệ
        # (Numpy đủ thông minh để tự broadcast mask 2D lên ảnh 3D hoặc 2D)
        image[mask] = 0.0

    return image

# =========================================================================

## image file I/O =================================

def writeImageFloat(image: np.ndarray, tiff_path: str, thumbnail=None):
    image = toNumpy(image)
    with tifffile.TiffWriter(tiff_path) as tiff:
        if thumbnail is not None:
            if thumbnail.dtype != np.uint8:
                thumbnail = thumbnail.astype(np.uint8)
            tiff.write(thumbnail, photometric='rgb', planarconfig='contig', bitspersample=8)

        if image.dtype != np.float32:
            image = image.astype(np.float32)
        tiff.write(image, photometric='minisblack', planarconfig='contig', bitspersample=32, compression='zlib')

def readImageFloat(tiff_path: str, return_thumbnail = False, read_or_die = True):
    try:
        multi_image = skimage.io.ImageCollection(tiff_path)
        num_read_images = len(multi_image)
        if num_read_images == 0:
            raise Exception('No images found.')
        elif num_read_images == 1:
            return multi_image[0].squeeze(), None
        elif num_read_images == 2: 
            if multi_image[0].dtype == np.uint8:
                if not return_thumbnail: return multi_image[1].squeeze()
                else: return multi_image[1].squeeze(), multi_image[0].squeeze()
            else:
                if not return_thumbnail: return multi_image[0].squeeze()
                else: return multi_image[0].squeeze(), multi_image[1].squeeze()
        else: 
            return [im.squeeze() for im in multi_image]       
    except Exception as e:
        LOG_ERROR('Failed to read image float: "%s"' %(e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None, None

def writeImage(image: np.ndarray, path: str):
    image = toNumpy(image)
    skimage.io.imsave(path, image)

def readImage(path: str, read_or_die=True):
    """
    Dùng OpenCV để đọc ảnh thay cho imageio (ổn định hơn)
    """
    try:
        image = cv2.imread(path)
        if image is None:
            raise Exception("cv2.imread returned None")
        # Chuyển BGR -> RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f'[ERROR] Failed to read image: "{path}" -> {e}')
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit(1)
        return None