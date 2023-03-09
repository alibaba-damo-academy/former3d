import inspect

import torch
from torch import nn
import spconv.pytorch as spconv

from .ops import former_utils


def formerTensor2spconvTensor(former_tensor, spconv_tensor=None):
    if spconv_tensor is None:
        spconvTensor = spconv.SparseConvTensor(
            former_tensor.features, bzyx2bxyz(former_tensor.indices), 
            former_tensor.spatial_shape, former_tensor.batch_size)
    else:
        spconvTensor = spconv.SparseConvTensor(
            former_tensor.features, bzyx2bxyz(former_tensor.indices), 
            former_tensor.spatial_shape, former_tensor.batch_size, 
            spconv_tensor.grid, spconv_tensor.voxel_num, spconv_tensor.indice_dict,
            spconv_tensor.benchmark)
    return spconvTensor


def spconvTensor2formerTensor(spconv_tensor, spvt_tensor):
    new_hash_size = int(spvt_tensor.hash_size / (spvt_tensor.spatial_shape[0] / spconv_tensor.spatial_shape[0])**3)
    formerTensor = SparseTensor(
            features = spconv_tensor.features,
            indices = bxyz2bzyx(spconv_tensor.indices),
            indice_dict = {**spconv_tensor.indice_dict, **spvt_tensor.indice_dict},
            spatial_shape = spconv_tensor.spatial_shape,
            voxel_size = None,
            point_cloud_range = None,
            batch_size = spconv_tensor.batch_size,
            hash_size = new_hash_size,
            map_table = None,
            gather_dict = None,
        )
    return formerTensor


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class SparseTensor(object):
    def __init__(self, features, indices, indice_dict, spatial_shape, voxel_size, point_cloud_range, batch_size, hash_size, map_table = None, gather_dict = None):
        self.features = features
        self.indices = indices
        self.indice_dict = indice_dict
        self.spatial_shape = spatial_shape # [x, y, z]
        self.batch_size = batch_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.hash_size = hash_size
        self.gather_dict = gather_dict
        self.map_table = self.build_map_table() if not map_table else map_table

    @torch.no_grad()
    def build_map_table(self):
        bs_cnt = torch.zeros(self.batch_size).int()
        for i in range(self.batch_size):
            bs_cnt[i] = (self.indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(self.indices.device)
        map_table = former_utils.build_hash_table(
            self.batch_size,
            self.hash_size,
            self.spatial_shape,
            self.indices,
            bs_cnt,
        )
        return map_table

    def dense(self, channels_first=True):
        reverse_spatial_shape = self.spatial_shape[::-1] # (ZYX)
        output_shape = [self.batch_size] + list(
            reverse_spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        ndim = len(reverse_spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

class Attention3d(nn.Module):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, norm):
        super(Attention3d, self).__init__()
        self.attention_modes = attention_modes

        self.mhead_attention = nn.MultiheadAttention(
                embed_dim= input_channels,
                num_heads= num_heads,
                dropout= dropout,
                )
        self.drop_out = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_channels, ff_channels)
        self.linear2 = nn.Linear(ff_channels, input_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

        self.output_layer = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            norm(output_channels),
            nn.ReLU(True)
        )

    @torch.no_grad()
    def with_bs_cnt(self, indices, batch_size):
        bs_cnt = torch.zeros(batch_size).int()
        for i in range(batch_size):
            bs_cnt[i] = (indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(indices.device)
        return bs_cnt

    @torch.no_grad()
    def with_coords(self, indices, point_cloud_range, voxel_size):
        voxel_size = torch.tensor(voxel_size).unsqueeze(0).to(indices.device)
        min_range = torch.tensor(point_cloud_range[0:3]).unsqueeze(0).to(indices.device)
        coords = (indices[:, [3, 2, 1]].float() + 0.5) * voxel_size + min_range
        return coords

    def forward(self, sp_tensor):
        raise NotImplementedError


class SubMAttention3d(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, norm,
                 use_pos_emb=True, use_relative_coords=True, use_no_query_coords=True, atten_sort=False):
        super(SubMAttention3d, self).__init__(input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, norm)

        self.use_relative_coords = use_relative_coords
        self.use_no_query_coords = use_no_query_coords
        self.use_pos_emb = use_pos_emb
        self.atten_sort = atten_sort

        self.norm1 = norm(input_channels)
        self.norm2 = norm(input_channels)
        if self.use_pos_emb:
            if not self.use_no_query_coords:
                self.q_pos_proj = nn.Sequential(
                    nn.Linear(3, input_channels),
                    nn.ReLU(True),
                )
            self.k_pos_proj = nn.Sequential(
                nn.Conv1d(3, input_channels, 1),
                nn.ReLU(True),
            )

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_range = attention_mode.RANGE
                _gather_indices = former_utils.subm_local_attention_hash_indices(spatial_shape, attend_size, attend_range, map_table, voxel_indices)
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                range_spec = attention_mode.RANGE_SPEC
                if self.atten_sort:
                    _gather_indices = former_utils.subm_strided_attention_hash_indices_sort(spatial_shape, attend_size, range_spec, map_table, voxel_indices)
                else:
                    _gather_indices = former_utils.subm_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, map_table, voxel_indices)
            else:
                raise NotImplementedError

            _gather_mask = (_gather_indices < 0)
            #_gather_indices[_gather_indices < 0] = 0
            _gather_dict[attention_mode.NAME] = [_gather_indices, _gather_mask]

        return _gather_dict

    def forward(self, sp_tensor):
        if not sp_tensor.gather_dict:
            sp_tensor.gather_dict = self.create_gather_dict(self.attention_modes, sp_tensor.map_table, sp_tensor.indices, sp_tensor.spatial_shape)

        voxel_features = sp_tensor.features
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = v_bs_cnt.clone()

        a_key_indices, a_key_mask = [], []
        for attention_idx, attetion_mode in enumerate(self.attention_modes):
            key_indices, key_mask = sp_tensor.gather_dict[attetion_mode.NAME]
            a_key_indices.append(key_indices)
            a_key_mask.append(key_mask)

        key_indices = torch.cat(a_key_indices, dim = 1)
        key_mask = torch.cat(a_key_mask, dim = 1)

        query_features = voxel_features.unsqueeze(0) # (1, N1+N2, C)
        if voxel_features.dtype == torch.float16:
            key_features = former_utils.grouping_operation(voxel_features.float(), v_bs_cnt, key_indices, k_bs_cnt).half()
        else:
            print("!!!!!!!! not fp16")
            key_features = former_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt)

        if self.use_pos_emb:
            # FIXME: position embedding use world coords or voxel coords?
            # voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
            voxel_coords = sp_tensor.indices[:, [3, 2, 1]].float()    # use voxel coords
            key_coords = former_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices, k_bs_cnt)
            # FIXME: cast to float16
            # if voxel_features.dtype == torch.float16:
            #     key_coords = key_coords.half()
            if self.use_relative_coords:
                key_coords = key_coords - voxel_coords.unsqueeze(-1)
            key_pos_emb = self.k_pos_proj(key_coords)
            key_features = key_features + key_pos_emb

            if self.use_no_query_coords:
                pass
            else:
                query_pos_emb = self.q_pos_proj(voxel_coords).unsqueeze(0)
                query_features = query_features + query_pos_emb

        key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)

        attend_features, attend_weights = self.mhead_attention(
            query = query_features,
            key = key_features,
            value = key_features,
            key_padding_mask = key_mask,
        )
        if torch.any(torch.isnan(attend_features)):
            print('!!! Warning: out of FP16 range, use FP32')
            with torch.cuda.amp.autocast(enabled=False):
                attend_features, attend_weights = self.mhead_attention(
                    query = query_features.float(),
                    key = key_features.float(),
                    value = key_features.float(),
                    key_padding_mask = key_mask,
                )
                attend_features = attend_features.clamp(-65500, 65500).half()

        if torch.any(torch.isnan(voxel_features)) or torch.any(torch.isnan(attend_features)):
            print('!!!!!!!!!!! Nan', voxel_features.max(), attend_features.max())

        attend_features = self.drop_out(attend_features)
        voxel_features = voxel_features + attend_features.squeeze(0)
        voxel_features = self.norm1(voxel_features)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(voxel_features))))
        voxel_features = voxel_features + self.dropout2(act_features)
        voxel_features = self.norm2(voxel_features)
        voxel_features = self.output_layer(voxel_features)
        sp_tensor.features = voxel_features
        return sp_tensor


class SubMAttenResBlock(nn.Module):
    def __init__(self, subm_cfg, norm, use_relative_coords=True, use_pooled_feature=True, use_no_query_coords=True, atten_sort=False):
        super(SubMAttenResBlock, self).__init__()
        self.subm_attention_modules = nn.ModuleList()
        for i in range(subm_cfg.NUM_BLOCKS):
            self.subm_attention_modules.append(SubMAttention3d(
                input_channels=subm_cfg.CHANNELS[0],
                output_channels=subm_cfg.CHANNELS[2],
                ff_channels=subm_cfg.CHANNELS[1],
                dropout=subm_cfg.DROPOUT,
                num_heads=subm_cfg.NUM_HEADS,
                attention_modes=subm_cfg.ATTENTION,
                norm=norm,
                use_pos_emb=subm_cfg.USE_POS_EMB,
                use_relative_coords=use_relative_coords,
                use_no_query_coords=use_no_query_coords,
                atten_sort=atten_sort
            ))

    def forward(self, sp_tensor):
        indentity_features = sp_tensor.features
        for subm_module in self.subm_attention_modules:
            sp_tensor = subm_module(sp_tensor)
        sp_tensor.features = sp_tensor.features + indentity_features
        return sp_tensor


class SparseConvTensor(object):
    def __init__(self, features, indices):
        self.features = features
        self.indices = indices


# former: bzyx
# spconv: bxyz
# torchsparse: xyzb

def xyzb2bxyz(indices):
    new_indices = torch.zeros_like(indices)
    new_indices[:, 0] = indices[:, 3]
    new_indices[:, 1] = indices[:, 0]
    new_indices[:, 2] = indices[:, 1]
    new_indices[:, 3] = indices[:, 2]
    return new_indices


def bxyz2xyzb(indices):
    new_indices = torch.zeros_like(indices)
    new_indices[:, 0] = indices[:, 1]
    new_indices[:, 1] = indices[:, 2]
    new_indices[:, 2] = indices[:, 3]
    new_indices[:, 3] = indices[:, 0]
    return new_indices


def xyzb2bzyx(indices):
    new_indices = torch.zeros_like(indices)
    new_indices[:, 0] = indices[:, 3]
    new_indices[:, 1] = indices[:, 2]
    new_indices[:, 2] = indices[:, 1]
    new_indices[:, 3] = indices[:, 0]
    return new_indices


def bzyx2xyzb(indices):
    return xyzb2bzyx(indices)


def bxyz2bzyx(indices):
    new_indices = torch.zeros_like(indices)
    new_indices[:, 0] = indices[:, 0]
    new_indices[:, 1] = indices[:, 3]
    new_indices[:, 2] = indices[:, 2]
    new_indices[:, 3] = indices[:, 1]
    return new_indices


def bzyx2bxyz(indices):
    return bxyz2bzyx(indices)


def combineSparseConvTensor(xs, device):
    features_batch = []
    indices_batch = []
    spatial_shape = xs[0].spatial_shape
    batch_size = len(xs)
    for i, x in enumerate(xs):
        features_batch.append(x.features.to(device))
        inds = x.indices
        inds[:, 0] = i
        indices_batch.append(inds.to(device))
    features_batch = torch.cat(features_batch, dim=0)
    indices_batch = torch.cat(indices_batch, dim=0)

    spconvTensor = spconv.SparseConvTensor(
        features_batch, indices_batch, 
        spatial_shape, batch_size)

    return spconvTensor


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


def sparsify(volume, valid, viz=False):

    shape = volume.shape[2:]
    B = volume.shape[0]
    X, Y, Z = volume.shape[-3:]

    valid_indices = torch.nonzero(valid[:, 0]).int().contiguous()
    features = volume.transpose(0, 1).flatten(1).transpose(0, 1)
    valid = valid.transpose(0, 1).flatten(1).transpose(0, 1)
    valid_features = features[torch.nonzero(valid[:, 0]).squeeze(1)]

    # sparse_volume = volume.permute(0,2,3,4,1).to_sparse(4)
    # values = sparse_volume.values()
    # indices = sparse_volume.indices().int().permute(1, 0).contiguous()
    # assert values.shape[0] == (valid > 0).sum(), "sparse values num == valid num"
    # assert torch.allclose(values, valid_features), "torch.allclose(values, valid_features)"
    # assert torch.allclose(indices, valid_indices), "torch.allclose(indices, valid_indices)"

    if viz:
        print("== Sparsity: ", valid_indices.shape[0]/valid.shape[0], volume.shape, (valid>0).sum().cpu().item())

    sparse_tensor = spconv.SparseConvTensor(valid_features, valid_indices, shape, B)

    return sparse_tensor


def global_avg_pool(inputs: spconv.SparseConvTensor) -> torch.Tensor:
    B = inputs.indices[-1][0] + 1
    outputs = []
    for b in range(B):
        input = inputs.features[inputs.indices[:, 0] == b]
        output = torch.mean(input, dim=0)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def autocast_norm(layer_class):
    class AutocastNorm(layer_class):
        def forward(self, input):
            if input.dtype == torch.float16:
                output = super().forward(input.float()).half()
            else:
                output = super().forward(input)
            return output
    return AutocastNorm


class SparseResNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=32):
        super().__init__()
        self.spconv1 = spconv.SparseSequential(
                spconv.SparseConv3d(input_dim, hidden_dim, 3, 1, 1), # just like nn.Conv3d but don't support group
                nn.BatchNorm1d(hidden_dim), # non-spatial layers can be used directly in SparseSequential.
                nn.ReLU())
        self.block1 = SparseBasicBlock(hidden_dim, hidden_dim, indice_key='block1')
        self.block2 = SparseBasicBlock(hidden_dim, hidden_dim, indice_key='block2')
        self.spconv2 = spconv.SparseSequential(
                spconv.SparseConv3d(hidden_dim, output_dim, 3, 1, 1),
                nn.BatchNorm1d(output_dim),
                nn.ReLU())

    def forward(self, x):
        x = self.spconv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.spconv2(x)
        return [x]# .dense()


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 indice_key=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        # out.features += identity.features
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)