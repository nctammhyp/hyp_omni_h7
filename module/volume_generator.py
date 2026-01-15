# File author: Hualie Jiang (jianghualie0@gmail.com)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out=1):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Conv3d(ch_hid, ch_out, (1, 1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out_act(x)
        return x


# class Generator(torch.nn.Module):
#     def __init__(self, opts):
#         super(Generator, self).__init__()
#         ch_in = opts.base_channel

#         # CŨ: 
#         # self.reference_mapping = MLP(2*ch_in+4, ch_in)
#         # self.target_mapping = MLP(2*ch_in+4, ch_in)

#         # MỚI: 
#         # Nếu Cam Reference chỉ dùng 1 cam (Cam 0), ta không cần trộn, bỏ reference_mapping.
#         # Nếu Cam Target dùng 2 cam (Cam 1 & 2), ta giữ target_mapping.
        
#         # Mapping trộn 2 camera mục tiêu (ví dụ Trái và Phải)
#         self.target_mapping = MLP(2*ch_in+4, ch_in) 

#     def forward(self, spherical_feats):
#         # spherical_feats bây giờ là list có 3 phần tử: [Feat0, Feat1, Feat2]
        
#         # --- XỬ LÝ REFERENCE (Cam 0) ---
#         # Vì chỉ có 1 cam phía trước, ta lấy trực tiếp làm Reference
#         reference_feat = spherical_feats[0] 

#         # --- XỬ LÝ TARGET (Cam 1 & 2) ---
#         # Trộn Cam 1 và Cam 2 dựa trên trọng số học được
#         # Concatenation của Feat1 và Feat2
#         target_feat_input = torch.cat([spherical_feats[1], spherical_feats[2]], dim=1)
        
#         # Tính trọng số trộn (Right/Left weight)
#         right_weight = self.target_mapping(target_feat_input)
        
#         # Trộn đặc trưng
#         target_feat = right_weight * spherical_feats[1] + (1 - right_weight) * spherical_feats[2]

#         context_feat = reference_feat

#         return [reference_feat, target_feat], context_feat


# File author: Hualie Jiang (jianghualie0@gmail.com)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out=1):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Conv3d(ch_hid, ch_out, (1, 1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out_act(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        ch_in = opts.base_channel

        # Mapping trộn 2 camera mục tiêu (ví dụ Trái và Phải)
        # Input size = (2 * ch_in) + (2 * 2_grid_coords) = 2*ch_in + 4
        self.target_mapping = MLP(2*ch_in+4, ch_in) 

    def forward(self, spherical_feats):
        # spherical_feats structure: [Feat0, Feat1, Feat2, Grid0, Grid1, Grid2]
        
        # --- XỬ LÝ REFERENCE (Cam 0) ---
        # Vì chỉ có 1 cam phía trước, ta lấy trực tiếp làm Reference
        reference_feat = spherical_feats[0] 

        # --- XỬ LÝ TARGET (Cam 1 & 2) ---
        # Tính trọng số trộn (Right/Left weight)
        # Cần nối cả Features và Grids để mạng học được sự phụ thuộc không gian
        target_feat_input = torch.cat([
            spherical_feats[1], 
            spherical_feats[2], 
            spherical_feats[4], 
            spherical_feats[5]
        ], dim=1)
        
        right_weight = self.target_mapping(target_feat_input)
        
        # Trộn đặc trưng
        target_feat = right_weight * spherical_feats[1] + (1 - right_weight) * spherical_feats[2]

        context_feat = reference_feat

        return [reference_feat, target_feat], context_feat