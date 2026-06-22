"""
VLM-guided Zone-of-Interest (ZOI) module -- Route 1: sparse relevance tokens.

N learned queries cross-attend (DETR-style) to a genuine high-resolution mid-stage
fused feature grid (e.g. the 32x32 TransfuserBackbone stage, NOT the 8x8 planner
bottleneck) and each query predicts a continuous (x, y) position in ego BEV meters
plus an importance score. The resulting N tokens are appended to the planner's
token memory (see model.py), so spatial precision lives in the continuous positional
encoding rather than being limited by the 8x8 grid's ~8m/cell resolution.

Train/inference parity: queries are learned and self-contained, consuming only
image/LiDAR features. There is no VLM at inference -- it only supplies the training
target for zoi_xy/zoi_imp (see the ZOI label generators in transfuser_zoi/). Closed-loop
CARLA runs are therefore unaffected by the VLM's absence at test time.
"""
import torch
from torch import nn

import transfuser_utils as t_u


class ContinuousPositionEmbedding(nn.Module):
  """
  Sinusoidal positional encoding for continuous (x, y) coordinates in ego meters.
  Analogous to PositionEmbeddingSine but for points instead of grid cells, so spatial
  precision is not quantized to a grid cell size.
  """

  def __init__(self, d_model, min_x, max_x, min_y, max_y, temperature=10000.0):
    super().__init__()
    assert d_model % 4 == 0, 'd_model must be divisible by 4 (x/y halves, each split into sin/cos).'
    self.center_x = (max_x + min_x) / 2.0
    self.range_x = (max_x - min_x) / 2.0
    self.center_y = (max_y + min_y) / 2.0
    self.range_y = (max_y - min_y) / 2.0

    num_pos_feats = d_model // 2
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    self.register_buffer('dim_t', dim_t, persistent=False)

  def forward(self, xy):
    """
    xy: [..., 2] continuous (x, y) coordinates in ego meters.
    returns: [..., d_model] positional encoding.
    """
    # Map ego meters to [0, 2*pi] so the sinusoid frequencies are meaningful across the BEV range.
    x_norm = (xy[..., 0] - self.center_x + self.range_x) / (2.0 * self.range_x) * (2.0 * torch.pi)
    y_norm = (xy[..., 1] - self.center_y + self.range_y) / (2.0 * self.range_y) * (2.0 * torch.pi)

    pos_x = x_norm.unsqueeze(-1) / self.dim_t
    pos_y = y_norm.unsqueeze(-1) / self.dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class ZoiModule(nn.Module):
  """
  DETR-style ZOI head. See module docstring for the overall design.

  forward(src_feature_grid) -> (zoi_tokens, zoi_xy, zoi_imp)
    src_feature_grid: [B, C_src, H, W] high-resolution fused feature map (config.zoi_src_stage).
    zoi_tokens: [B, N, d_model] -- ready to concatenate onto the planner token memory.
    zoi_xy:     [B, N, 2]       -- predicted (x, y) in ego BEV meters (continuous).
    zoi_imp:    [B, N]          -- raw importance logits (pre-sigmoid; for the BCE loss).
  """

  def __init__(self, config, src_channels):
    super().__init__()
    self.config = config
    d_model = config.gru_input_size

    self.input_proj = nn.Conv2d(src_channels, d_model, kernel_size=1)
    self.src_pos_encoding = t_u.PositionEmbeddingSine(d_model // 2, normalize=True)

    self.query_embed = nn.Parameter(torch.zeros(1, config.zoi_num_queries, d_model))
    nn.init.uniform_(self.query_embed)

    decoder_layer = nn.TransformerDecoderLayer(d_model,
                                               config.zoi_num_heads,
                                               activation=nn.GELU(),
                                               batch_first=True)
    self.decoder = nn.TransformerDecoder(decoder_layer,
                                         num_layers=config.zoi_num_decoder_layers,
                                         norm=nn.LayerNorm(d_model))

    self.xy_head = nn.Linear(d_model, 2)
    self.importance_head = nn.Linear(d_model, 1)
    self.pos_encoding_xy = ContinuousPositionEmbedding(d_model, config.min_x, config.max_x, config.min_y,
                                                       config.max_y)
    self.token_norm = nn.LayerNorm(d_model)

  def predict_xy(self, content):
    raw = self.xy_head(content)
    x = torch.tanh(raw[..., 0]) * self.pos_encoding_xy.range_x + self.pos_encoding_xy.center_x
    y = torch.tanh(raw[..., 1]) * self.pos_encoding_xy.range_y + self.pos_encoding_xy.center_y
    return torch.stack((x, y), dim=-1)

  def forward(self, src_feature_grid):
    bs = src_feature_grid.shape[0]

    memory = self.input_proj(src_feature_grid)
    memory = memory + self.src_pos_encoding(memory)
    memory = torch.flatten(memory, start_dim=2).permute(0, 2, 1)  # [B, H*W, d_model]

    queries = self.query_embed.repeat(bs, 1, 1)
    content = self.decoder(queries, memory)  # [B, N, d_model]

    zoi_xy = self.predict_xy(content)
    zoi_imp = self.importance_head(content).squeeze(-1)  # [B, N] raw logits

    token = self.token_norm(content + self.pos_encoding_xy(zoi_xy))
    zoi_tokens = token * (1.0 + torch.sigmoid(zoi_imp)).unsqueeze(-1)

    return zoi_tokens, zoi_xy, zoi_imp
