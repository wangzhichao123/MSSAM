# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type

from .common import LayerNorm2d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MSFG(nn.Module):
    def __init__(self, dim=768, scale=4):
        super(MSFG, self).__init__()

        self.conv3 = BasicConv2d(dim, dim // scale, 1)
        self.conv6 = BasicConv2d(dim, dim // scale, 1)
        self.conv9 = BasicConv2d(dim, dim // scale, 1)
        self.conv12 = BasicConv2d(dim, dim // scale, 1)

        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = BasicConv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv_3 = BasicConv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv_4 = BasicConv2d(dim, dim, kernel_size=3, padding=1, stride=1)

    def forward(self, sam_list):

        x3 = sam_list[0].permute(0, 3, 1, 2)  # [B, dim, 32, 32]
        x6 = sam_list[1].permute(0, 3, 1, 2)  # [B, dim, 32, 32]
        x9 = sam_list[2].permute(0, 3, 1, 2)  # [B, dim, 32, 32]
        x12 = sam_list[3].permute(0, 3, 1, 2)  # [B, dim, 32, 32]

        x3 = self.conv3(x3)
        x6 = self.conv6(x6)
        x9 = self.conv9(x9)
        x12 = self.conv12(x12)

        H1 = torch.cat([x3, x6, x9, x12], dim=1)   # [B, dim, 32, 32]

        H_c = self.ca(H1) * H1  # channel attention
        H_c_s = self.sa(H_c) * H_c  # spatial attention

        H2 = self.conv_2(self.max_pool(H1))       # [B, dim, 16, 16]
        H3 = self.conv_3(self.max_pool(H2))      # [B, dim, 8, 8]
        H4 = self.conv_3(self.max_pool(H3))      # [B, dim, 4, 4]

        return [H1, H2, H3, H4]

class CA(nn.Module):
    def __init__(self, in_planes, mid_planes, ratio=16):
        super(CA, self).__init__()
        self.conv = nn.Conv2d(2 * in_planes + mid_planes, mid_planes, 1, 1, 0)
        self.bn = nn.BatchNorm2d(mid_planes)
        self.SiLU = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(mid_planes, mid_planes // ratio, 1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(mid_planes // ratio, mid_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)  # 3->1
        x = self.bn(x)
        x = self.SiLU(x)
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        out = x * attn + res
        return out

class MSFD(nn.Module):
    def __init__(self, dim=768):
        super(MSFD, self).__init__()

        self.ca0 = CA(dim, 256)
        self.ca1 = CA(dim, 256)
        self.ca2 = CA(dim, 256)
        self.ca3 = CA(dim, 256)

    def forward(self, sam_list, msfg_list, transformer_feature_0):

        s0 = sam_list[0].permute(0, 3, 1, 2)  # [B, dim, 32, 32]
        s1 = sam_list[1].permute(0, 3, 1, 2)  # [B, dim, 32, 32]
        s2 = sam_list[2].permute(0, 3, 1, 2)  # [B, dim, 32, 32]
        s3 = sam_list[3].permute(0, 3, 1, 2)  # [B, dim, 32, 32]

        m0 = msfg_list[0]  # [B, dim, 32, 32]
        m1 = msfg_list[1]  # [B, dim, 16, 16]
        m2 = msfg_list[2]  # [B, dim, 8, 8]
        m3 = msfg_list[3]  # [B, dim, 4, 4]

        f0 = torch.cat([s0, m0, transformer_feature_0], dim=1)   # [B, 3 * dim, 32, 32]
        transformer_feature_1 = self.ca0(f0)   # [B, dim, 32, 32]

        m1 = F.interpolate(m1, scale_factor=2, mode='bilinear')
        f1 = torch.cat([s1, m1, transformer_feature_1], dim=1)  # [B, 3 * dim, 32, 32]
        transformer_feature_2 = self.ca1(f1)  # [B, dim, 32, 32]

        m2 = F.interpolate(m2, scale_factor=4, mode='bilinear')
        f2 = torch.cat([s2, m2, transformer_feature_2], dim=1)  # [B, 3 * dim, 32, 32]
        transformer_feature_3 = self.ca2(f2)  # [B, dim, 32, 32]

        m3 = F.interpolate(m3, scale_factor=8, mode='bilinear')
        f3 = torch.cat([s3, m3, transformer_feature_3], dim=1)  # [B, 3 * dim, 32, 32]
        transformer_feature_4 = self.ca3(f3)  # [B, dim, 32, 32]

        return transformer_feature_4

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        
        self.msfg = MSFG(768, 4)
        self.msfd = MSFD(768)
        self.segHead = nn.Conv2d(transformer_dim, num_multimask_outputs, 1, 1, 0)

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        # don't change
        # groups=transformer_dim // 8
        self.cls_upscaling = nn.Sequential(
            # nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2),
            # LayerNorm2d(transformer_dim // 4),
            # activation(),
            # nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=7, stride=2, padding=3),
            # activation(),
            # nn.Conv2d(transformer_dim // 8, transformer_dim // 4, kernel_size=1, stride=1, padding=0, bias=True),
            # activation(),
            # nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 16, kernel_size=1, stride=1, padding=0, bias=True),
            activation(),
            nn.Conv2d(transformer_dim // 16, transformer_dim // 8, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sam_list: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        
        # diy decoder
        msfg_list = self.msfg(sam_list)
        diy_decoder_feature = self.msfd(sam_list, msfg_list, image_embeddings)
        diy_decoder_mask = self.segHead(diy_decoder_feature)
        diy_decoder_mask = F.interpolate(diy_decoder_mask, scale_factor=4, mode='bilinear')

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred, diy_decoder_mask

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        src_feature = src
        
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        upscaled_embedding_src = self.output_upscaling(src_feature)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]
        
        # upscaled_embedding_concat = torch.cat([upscaled_embedding, upscaled_embedding_src], dim=1)  # add
        upscaled_embedding_concat = upscaled_embedding + upscaled_embedding_src
        cls_upscaled_embedding = self.cls_upscaling(upscaled_embedding_concat)
        
        b, c, h, w = cls_upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
