import numpy as np
import torch


class ViewDirectionEncoder(torch.nn.Module):
    def __init__(self, feat_depth, L):
        super().__init__()
        self.L = L
        self.view_embedding_dim = 3 + self.L * 6
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                feat_depth + self.view_embedding_dim, feat_depth, 1, bias=False
            ),
        )
        torch.nn.init.xavier_normal_(self.conv[0].weight)

    def forward(self, feats, proj, cam_positions):
        device = feats.device
        dtype = feats.dtype
        featheight, featwidth = feats.shape[2:]
        u = torch.arange(featwidth, device=device, dtype=dtype)
        v = torch.arange(featheight, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        ones = torch.ones_like(uu)
        uv = torch.stack((uu, vv, ones, ones), dim=0).to(dtype)

        inv_proj = torch.linalg.inv(proj)
        xyz = inv_proj @ uv.reshape(4, -1)
        view_vecs = xyz[:, :, :3] - cam_positions[..., None]
        view_vecs /= torch.linalg.norm(view_vecs, dim=2, keepdim=True)
        view_vecs = view_vecs.to(dtype)

        view_encoding = [view_vecs]
        for i in range(self.L):
            view_encoding.append(torch.sin(view_vecs * np.pi * 2 ** i))
            view_encoding.append(torch.cos(view_vecs * np.pi * 2 ** i))
        view_encoding = torch.cat(view_encoding, dim=2)

        view_encoding = view_encoding.reshape(
            view_encoding.shape[0] * view_encoding.shape[1],
            view_encoding.shape[2],
            featheight,
            featwidth,
        )

        feats = torch.cat((feats, view_encoding), dim=1)
        feats = self.conv(feats)
        return feats
