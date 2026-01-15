# adapted from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/corr.py
import torch
import torch.nn.functional as F


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        Args:
            fmap1, fmap2: feature maps with shape [B, C, H, W, D]
            num_levels: number of levels in correlation pyramid
            radius: neighborhood radius for correlation sampling
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all-pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        bs, ch, h, w, nd = fmap1.shape
        self.bs = bs
        self.h = h
        self.w = w
        self.nd = nd

        corr = corr.reshape(bs * h * w, 1, 1, nd)
        self.corr_pyramid.append(corr)

        for i in range(1, self.num_levels):
            # Downsample correlation for pyramid
            corr = F.avg_pool2d(corr, kernel_size=(1, 2), stride=(1, 2))
            self.corr_pyramid.append(corr)

    def __call__(self, invdepth_idx):
        """
        Sample correlations around predicted inverse depth indices.
        Args:
            invdepth_idx: predicted indices, shape [B, 1, H, W]
        Returns:
            sampled correlations: [B, C_out, H, W]
        """
        # Clamp invdepth indices to valid range
        invdepth_idx = torch.clamp(invdepth_idx, 0, self.nd - 1).float()

        r = self.radius
        coords = invdepth_idx.permute(0, 2, 3, 1)  # [B, H, W, 1]
        batch, h, w, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # [B*H*W, 1, 1, D_lvl]

            # 1D sampling around radius
            dx = torch.linspace(-r, r, 2*r + 1, device=coords.device).view(2*r + 1, 1)

            # compute grid coordinates in [-1, 1] for grid_sample
            x0 = coords.reshape(batch*h*w, 1, 1, 1) / (2 ** i) + dx  # local offset
            x0 = 2.0 * torch.clamp(x0, 0, corr.shape[-1]-1) / (corr.shape[-1]-1) - 1.0
            y0 = torch.zeros_like(x0)

            # grid_sample expects [N, H_out, W_out, 2]
            coords_lvl = torch.cat([x0, y0], dim=-1)
            samp_corr = F.grid_sample(
                corr,
                coords_lvl,
                align_corners=True,
                mode='bilinear'
            )

            # reshape back to [B, H, W, C]
            samp_corr = samp_corr.view(batch, h, w, -1)
            out_pyramid.append(samp_corr)

        # concatenate all pyramid levels
        out = torch.cat(out_pyramid, dim=-1)  # [B, H, W, C_total]
        return out.permute(0, 3, 1, 2).contiguous()  # [B, C_total, H, W]

    @staticmethod
    def corr(fmap1, fmap2):
        """
        Compute all-pairs correlation.
        Args:
            fmap1, fmap2: [B, C, H, W, D]
        Returns:
            correlation volume: [B, H, W, 1, D]
        """
        assert fmap1.shape == fmap2.shape
        bs, ch, h, w, nd = fmap1.shape
        corr = torch.einsum('aijkh,aijkh->ajkh', fmap1, fmap2)
        corr = corr.reshape(bs, h, w, 1, nd).contiguous()
        return corr / torch.sqrt(torch.tensor(ch).float())
