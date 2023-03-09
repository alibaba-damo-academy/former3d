import collections
import numpy_indexed as npi

import torch
import spconv.pytorch as spconv
from spconv.pytorch.functional import _indice_to_scalar

from former3d import cnn2d, mv_fusion, utils, view_direction_encoder
from former3d.net3d.former_v1 import Former3D as backbone3d
from former3d.net3d.sparse3d import combineSparseConvTensor, xyzb2bxyz, bxyz2xyzb


class SDFFormer(torch.nn.Module):
    def __init__(self, attn_heads, attn_layers, use_proj_occ, voxel_size=0.04):
        super().__init__()
        self.use_proj_occ = use_proj_occ
        self.n_attn_heads = attn_heads
        self.resolutions = collections.OrderedDict(
            [
                ["coarse", voxel_size*4],
                ["medium", voxel_size*2],
                ["fine", voxel_size],
            ]
        )

        net2d_output_depths = [80, 40, 24]
        net3d_hidden_depths = [128, 64, 64]
        net3d_output_depths = [96, 48, 16]
        net3d_channels = [
            [128, 256, 512],
            [64, 128, 256],
            [64, 64, 128, 256, 512]
        ]

        self.net2d = cnn2d.MnasMulti(net2d_output_depths, pretrained=True)
        self.upsampler = Upsampler()

        self.output_layers = torch.nn.ModuleDict()
        self.net3d = torch.nn.ModuleDict()
        self.view_embedders = torch.nn.ModuleDict()
        self.layer_norms = torch.nn.ModuleDict()
        self.mv_fusion = torch.nn.ModuleDict()
        prev_output_depth = 0
        for i, (resname, res) in enumerate(self.resolutions.items()):
            self.view_embedders[resname] = view_direction_encoder.ViewDirectionEncoder(
                net2d_output_depths[i], L=4
            )
            self.layer_norms[resname] = torch.nn.LayerNorm(net2d_output_depths[i]*2)

            if self.n_attn_heads > 0:
                self.mv_fusion[resname] = mv_fusion.MVFusionTransformer(
                    net2d_output_depths[i], attn_layers, self.n_attn_heads, use_var=True
                )
            else:
                self.mv_fusion[resname] = mv_fusion.MVFusionMean()

            input_depth = prev_output_depth + net2d_output_depths[i] * 2
            if i > 0:
                # additional channel for the previous level's occupancy prediction
                input_depth += 1
            if resname == 'fine':
                net = backbone3d(input_depth=input_depth, channels=net3d_channels[i], post_deform=True,
                            hidden_depth=net3d_hidden_depths[i], output_depth=net3d_output_depths[i])
            else:
                net = backbone3d(input_depth=input_depth, channels=net3d_channels[i], post_deform=False,
                            hidden_depth=net3d_hidden_depths[i], output_depth=net3d_output_depths[i])
            output_depth = net.output_depth
            self.net3d[resname] = net
            self.output_layers[resname] = spconv.SubMConv3d(output_depth, 1, 1, 1, padding=1, bias=True)
            prev_output_depth = net.output_depth

    def get_img_feats(self, rgb_imgs, proj_mats, cam_positions):
        batchsize, n_imgs, _, imheight, imwidth = rgb_imgs.shape
        feats = self.net2d(rgb_imgs.reshape((batchsize * n_imgs, *rgb_imgs.shape[2:])))
        for resname in self.resolutions:
            f = feats[resname]
            f = self.view_embedders[resname](f, proj_mats[resname], cam_positions)
            f = f.reshape((batchsize, n_imgs, *f.shape[1:]))
            feats[resname] = f
        return feats

    def forward(self, batch, voxel_inds_16):
        feats_2d = self.get_img_feats(
            batch["rgb_imgs"], batch["proj_mats"], batch["cam_positions"]
        )
        batch_size = batch["rgb_imgs"].shape[0]

        device = voxel_inds_16.device
        proj_occ_logits = {}
        voxel_outputs = {}
        bp_data = {}
        n_subsample = {
            "medium": 2 ** 14,
            "fine": 2 ** 16,
        }

        voxel_inds = voxel_inds_16
        voxel_dim_16 = voxel_inds_16[-1][:3] + 1
        voxel_features = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        voxel_logits = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        for resname, res in self.resolutions.items():
            if self.training and resname in n_subsample:
                # this saves memory and possibly acts as a data augmentation
                subsample_inds = get_subsample_inds(voxel_inds, n_subsample[resname])
                voxel_inds = voxel_inds[subsample_inds]
                voxel_features = voxel_features[subsample_inds]
                voxel_logits = voxel_logits[subsample_inds]

            voxel_batch_inds = voxel_inds[:, 3].long()
            voxel_coords = voxel_inds[:, :3] * res + batch["origin"][voxel_batch_inds]

            featheight, featwidth = feats_2d[resname].shape[-2:]
            bp_uv, bp_depth, bp_mask = self.project_voxels(
                voxel_coords,
                voxel_batch_inds,
                batch["proj_mats"][resname].transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data[resname] = {
                "voxel_coords": voxel_coords,
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }
            bp_feats, cur_proj_occ_logits = self.back_project_features(
                bp_data[resname],
                feats_2d[resname].transpose(0, 1),
                self.mv_fusion[resname],
            )
            proj_occ_logits[resname] = cur_proj_occ_logits

            bp_feats = self.layer_norms[resname](bp_feats)

            voxel_features = torch.cat((voxel_features, bp_feats, voxel_logits), dim=-1)

            hash_size = 20*n_subsample[resname] if resname in n_subsample else n_subsample['fine']
            voxel_dim = (voxel_dim_16*(self.resolutions['coarse']/res)).int().tolist()
            voxel_features = spconv.SparseConvTensor(voxel_features, xyzb2bxyz(voxel_inds), voxel_dim, batch_size)
            voxel_features = self.net3d[resname](voxel_features, voxel_dim, res, hash_size=hash_size)

            voxel_logits = self.output_layers[resname](voxel_features)
            voxel_outputs[resname] = voxel_logits

            if resname in ["coarse", "medium"]:
                # sparsify & upsample
                occupancy = voxel_logits.features.squeeze(1) > 0
                if not torch.any(occupancy):
                    return voxel_outputs, proj_occ_logits, bp_data
                voxel_features = self.upsampler.upsample_feats(
                    voxel_features.features[occupancy]
                )
                voxel_inds = self.upsampler.upsample_inds(bxyz2xyzb(voxel_logits.indices)[occupancy])
                voxel_logits = self.upsampler.upsample_feats(voxel_logits.features[occupancy])

        return voxel_outputs, proj_occ_logits, bp_data

    def losses(self, voxel_logits, voxel_gt, proj_occ_logits, bp_data, depth_imgs):
        voxel_losses = {}
        proj_occ_losses = {}
        for resname in voxel_logits:
            logits = voxel_logits[resname]
            gt = voxel_gt[resname]
            gt = combineSparseConvTensor(gt, device=logits.features.device)
            cur_loss = torch.zeros(1, device=logits.features.device, dtype=torch.float32)
            if len(logits.indices) > 0:
                pred_scalar = _indice_to_scalar(logits.indices, [logits.batch_size] + logits.spatial_shape)
                gt_scalar = _indice_to_scalar(gt.indices, [logits.batch_size] + logits.spatial_shape)
                idx_query = npi.indices(gt_scalar.cpu().numpy(), pred_scalar.cpu().numpy(), missing=-1)
                good_query = idx_query != -1

                gt = gt.features.squeeze(1)[idx_query[good_query]]
                logits = logits.features.squeeze(1)[good_query]
                if len(logits) > 0:
                    if resname == "fine":
                        cur_loss = torch.nn.functional.l1_loss(
                            utils.log_transform(1.05 * torch.tanh(logits)),
                            utils.log_transform(gt),
                        )
                    else:
                        cur_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, gt
                        )
                    voxel_losses[resname] = cur_loss

            proj_occ_losses[resname] = compute_proj_occ_loss(
                proj_occ_logits[resname],
                depth_imgs,
                bp_data[resname],
                truncation_distance=3 * self.resolutions[resname],
            )

        loss = sum(voxel_losses.values()) + sum(proj_occ_losses.values())
        logs = {
            **{
                f"voxel_loss_{resname}": voxel_losses[resname].item()
                for resname in voxel_losses
            },
            **{
                f"proj_occ_loss_{resname}": proj_occ_losses[resname].item()
                for resname in proj_occ_losses
            },
        }
        return loss, logs

    def project_voxels(
        self, voxel_coords, voxel_batch_inds, projmat, imheight, imwidth
    ):
        device = voxel_coords.device
        n_voxels = len(voxel_coords)
        n_imgs = len(projmat)
        bp_uv = torch.zeros((n_imgs, n_voxels, 2), device=device, dtype=torch.float32)
        bp_depth = torch.zeros((n_imgs, n_voxels), device=device, dtype=torch.float32)
        bp_mask = torch.zeros((n_imgs, n_voxels), device=device, dtype=torch.bool)
        batch_inds = torch.unique(voxel_batch_inds)
        for batch_ind in batch_inds:
            batch_mask = voxel_batch_inds == batch_ind
            if torch.sum(batch_mask) == 0:
                continue
            cur_voxel_coords = voxel_coords[batch_mask]

            ones = torch.ones(
                (len(cur_voxel_coords), 1), device=device, dtype=torch.float32
            )
            voxel_coords_h = torch.cat((cur_voxel_coords, ones), dim=-1)

            im_p = projmat[:, batch_ind] @ voxel_coords_h.t()
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x = im_x / im_z
            im_y = im_y / im_z
            im_grid = torch.stack(
                [2 * im_x / (imwidth - 1) - 1, 2 * im_y / (imheight - 1) - 1],
                dim=-1,
            )
            im_grid[torch.isinf(im_grid)] = -2
            mask = im_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

            bp_uv[:, batch_mask] = im_grid.to(bp_uv.dtype)
            bp_depth[:, batch_mask] = im_z.to(bp_uv.dtype)
            bp_mask[:, batch_mask] = mask

        return bp_uv, bp_depth, bp_mask

    def back_project_features(self, bp_data, feats, mv_fuser):
        n_imgs, batch_size, in_channels, featheight, featwidth = feats.shape
        device = feats.device
        n_voxels = len(bp_data["voxel_batch_inds"])
        feature_volume_all = torch.zeros(
            n_voxels, in_channels*2, device=device, dtype=torch.float32
        )
        # the default proj occ prediction is true everywhere -> logits high
        proj_occ_logits = torch.full(
            (n_imgs, n_voxels), 100, device=device, dtype=feats.dtype
        )
        batch_inds = torch.unique(bp_data["voxel_batch_inds"])
        for batch_ind in batch_inds:
            batch_mask = bp_data["voxel_batch_inds"] == batch_ind
            if torch.sum(batch_mask) == 0:
                continue

            cur_bp_uv = bp_data["bp_uv"][:, batch_mask]
            cur_bp_depth = bp_data["bp_depth"][:, batch_mask]
            cur_bp_mask = bp_data["bp_mask"][:, batch_mask]
            cur_feats = feats[:, batch_ind].view(
                n_imgs, in_channels, featheight, featwidth
            )
            cur_bp_uv = cur_bp_uv.view(n_imgs, 1, -1, 2)
            features = torch.nn.functional.grid_sample(
                cur_feats,
                cur_bp_uv.to(cur_feats.dtype),
                padding_mode="reflection",
                align_corners=True,
            )
            features = features.view(n_imgs, in_channels, -1)
            var_imgs = ((features-features.mean(dim=0))**2)
            var = var_imgs.mean(0)
            if isinstance(mv_fuser, mv_fusion.MVFusionTransformer):
                pooled_features, cur_proj_occ_logits = mv_fuser(
                    features,
                    cur_bp_depth,
                    cur_bp_mask,
                    self.use_proj_occ,
                    var_imgs,
                )
                proj_occ_logits[:, batch_mask] = cur_proj_occ_logits
            else:
                pooled_features = mv_fuser(features.transpose(1, 2), cur_bp_mask)
            # feature_volume_all[batch_mask] = pooled_features
            feature_volume_all[batch_mask] = torch.cat([pooled_features, var.transpose(0, 1)], dim=1)

        return (feature_volume_all, proj_occ_logits)


class Upsampler(torch.nn.Module):
    # nearest neighbor 2x upsampling for sparse 3D array

    def __init__(self):
        super().__init__()
        self.upsample_offsets = torch.nn.Parameter(
            torch.Tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 1, 0],
                    ]
                ]
            ).to(torch.int32),
            requires_grad=False,
        )
        self.upsample_mul = torch.nn.Parameter(
            torch.Tensor([[[2, 2, 2, 1]]]).to(torch.int32), requires_grad=False
        )

    def upsample_inds(self, voxel_inds):
        return (
            voxel_inds[:, None] * self.upsample_mul + self.upsample_offsets
        ).reshape(-1, 4)

    def upsample_feats(self, feats):
        return (
            feats[:, None]
            .repeat(1, 8, 1)
            .reshape(-1, feats.shape[-1])
            .to(torch.float32)
        )


def get_subsample_inds(coords, max_per_example):
    keep_inds = []
    batch_inds = coords[:, 3].unique()
    for batch_ind in batch_inds:
        batch_mask = coords[:, -1] == batch_ind
        n = torch.sum(batch_mask)
        if n > max_per_example:
            keep_inds.append(batch_mask.float().multinomial(max_per_example))
        else:
            keep_inds.append(torch.where(batch_mask)[0])
    subsample_inds = torch.cat(keep_inds).long()
    return subsample_inds


def compute_proj_occ_loss(proj_occ_logits, depth_imgs, bp_data, truncation_distance):
    batch_inds = torch.unique(bp_data["voxel_batch_inds"])
    for batch_ind in batch_inds:
        batch_mask = bp_data["voxel_batch_inds"] == batch_ind
        cur_bp_uv = bp_data["bp_uv"][:, batch_mask]
        cur_bp_depth = bp_data["bp_depth"][:, batch_mask]
        cur_bp_mask = bp_data["bp_mask"][:, batch_mask]
        cur_proj_occ_logits = proj_occ_logits[:, batch_mask]

        depth = torch.nn.functional.grid_sample(
            depth_imgs[batch_ind, :, None],
            cur_bp_uv[:, None].to(depth_imgs.dtype),
            padding_mode="zeros",
            mode="nearest",
            align_corners=False,
        )[:, 0, 0]

        proj_occ_mask = cur_bp_mask & (depth > 0)
        if torch.sum(proj_occ_mask) > 0:
            proj_occ_gt = torch.abs(cur_bp_depth - depth) < truncation_distance
            return torch.nn.functional.binary_cross_entropy_with_logits(
                cur_proj_occ_logits[proj_occ_mask],
                proj_occ_gt[proj_occ_mask].to(cur_proj_occ_logits.dtype),
            )
        else:
            return torch.zeros((), dtype=torch.float32, device=depth_imgs.device)
