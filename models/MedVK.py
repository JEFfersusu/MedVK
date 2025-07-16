import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=0.3,
            scale_spline=0.1,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
    @property
    def scaled_spline_weight(self):

        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):

        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.contiguous().view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

class KANBlock(nn.Module):

    def __init__(self, dim, depth=1, num_heads=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            KANformerBlock(dim, num_heads=num_heads) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DynamicRouting(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gate = KANLinear(dim, dim, grid_size=3)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, res):
        gate = torch.sigmoid(self.gate(self.norm(x)))
        return x * gate + res * (1 - gate)


class SpatialKAN(nn.Module):

    def __init__(self, dim, grid_size=5, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads


        self.to_qkv = KANLinear(dim, 3 * dim, grid_size=grid_size)
        self.scale = nn.Parameter(torch.tensor(1.0 / (self.head_dim ** 0.5)))

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.to_qkv(x)  # [B, H, W, 3*C]
        qkv = qkv.view(B, H * W, self.num_heads, 3 * self.head_dim)


        q, k, v = qkv.chunk(3, dim=-1)
        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            output = F.scaled_dot_product_attention(q, k, v)

        return output.permute(0, 2, 1, 3).contiguous().view(B, H, W, C)


class ChannelKAN(nn.Module):

    def __init__(self, dim, expansion=2):
        super().__init__()
        self.integrate = nn.Sequential(
            KANLinear(dim, dim * expansion, grid_size=5),
            nn.GELU(),
            KANLinear(dim * expansion, dim, grid_size=3)
        )

    def forward(self, x):
        return self.integrate(x)


class KANformerBlock(nn.Module):

    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_kan = SpatialKAN(dim, num_heads=num_heads)
        self.spatial_route = DynamicRouting(dim)

        self.channel_norm = nn.LayerNorm(dim)
        self.channel_kan = ChannelKAN(dim)
        self.channel_route = DynamicRouting(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        identity = x
        x = self.spatial_norm(x)
        x = self.spatial_kan(x)
        x = self.spatial_route(x, identity)

        identity = x
        x = self.channel_norm(x)
        x = self.channel_kan(x)
        x = self.channel_route(x, identity)

        return x.permute(0, 3, 1, 2)


class KANFusion(nn.Module):

    def __init__(self, feat_dims, target_dim):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_c, target_dim // 4, 1),
                nn.Flatten(start_dim=2)
            ) for in_c in feat_dims
        ])

        self.fuser = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.ReLU(),
            KANLinear(target_dim, target_dim, grid_size=3)
        )

    def forward(self, features):
        aligned = []
        for feat, adapter in zip(features, self.adapters):
            aligned_feat = adapter(feat).squeeze(-1)  # [B, target_dim//4]
            aligned.append(aligned_feat)

        fused = torch.cat(aligned, dim=1)  # [B, target_dim]
        return self.fuser(fused)


class MedVK_T(nn.Module):
    def __init__(self, image_size=224, base_dim=32, depth=[1, 1, 1, 1], num_classes=2):
        super().__init__()
        if len(depth) < 4:
            depth = list(depth) + [depth[-1]] * (4 - len(depth)) if depth else [2] * 4

        self.stages = nn.ModuleList()
        in_chans = 3

        stage_cfg = [
            (2, 2),  # Stage1: 112x112 (dim=16)
            (2, 2),  # Stage2: 56x56  (dim=16)
            (3, 2),  # Stage3: 28x28  (dim=24)
            (4, 1)  # Stage4: 28x28  (dim=32)
        ]

        for mult, stride in stage_cfg:
            out_dim = base_dim * mult


            stage = nn.Sequential(
                nn.Conv2d(in_chans, out_dim, 3, stride, 1),
                KANBlock(out_dim, depth=depth.pop(0), num_heads=1),
                nn.MaxPool2d(3, 2, 1) if stride > 1 else nn.Identity()
            )
            self.stages.append(stage)
            in_chans = out_dim

        self.fusion = KANFusion(
            feat_dims=[base_dim * 2, base_dim * 2, base_dim * 3, base_dim * 4],
            target_dim=64
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            # print(x.shape)
            features.append(x)
        x = self.fusion(features)

        x = self.fc(x)
        return x


class MedVK_S(nn.Module):
    def __init__(self, image_size=224, base_dim=48, depth=[1, 2, 1, 1], num_classes=2):
        super().__init__()
        if len(depth) < 4:
            depth = list(depth) + [depth[-1]] * (4 - len(depth)) if depth else [2] * 4

        self.stages = nn.ModuleList()
        in_chans = 3

        stage_cfg = [
            (2, 2),  # Stage1: 112x112 (dim=16)
            (2, 2),  # Stage2: 56x56  (dim=16)
            (4, 2),  # Stage3: 28x28  (dim=24)
            (4, 1)  # Stage4: 28x28  (dim=32)
        ]

        for mult, stride in stage_cfg:
            out_dim = base_dim * mult


            stage = nn.Sequential(
                nn.Conv2d(in_chans, out_dim, 3, stride, 1),
                KANBlock(out_dim, depth=depth.pop(0), num_heads=1),
                nn.MaxPool2d(3, 2, 1) if stride > 1 else nn.Identity()
            )
            self.stages.append(stage)
            in_chans = out_dim

        self.fusion = KANFusion(
            feat_dims=[base_dim * 2, base_dim * 2, base_dim * 4, base_dim * 4],
            target_dim=96
        )

        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            # print(x.shape)
        # features = [feat for feat in features if feat is not None]

        x = self.fusion(features)

        x = self.fc(x)
        # print(x,x.shape)
        return x


class MedVK_B(nn.Module):
    def __init__(self, image_size=224, base_dim=48, depth=[2, 3, 3, 1], num_classes=2):
        super().__init__()
        if len(depth) < 4:
            depth = list(depth) + [depth[-1]] * (4 - len(depth)) if depth else [2] * 4

        self.stages = nn.ModuleList()
        in_chans = 3

        stage_cfg = [
            (2, 2),  # Stage1: 112x112 (dim=16)
            (3, 2),  # Stage2: 56x56  (dim=16)
            (4, 2),  # Stage3: 28x28  (dim=24)
            (5, 1)  # Stage4: 28x28  (dim=32)
        ]

        for mult, stride in stage_cfg:
            out_dim = base_dim * mult

            stage = nn.Sequential(
                nn.Conv2d(in_chans, out_dim, 3, stride, 1),
                KANBlock(out_dim, depth=depth.pop(0), num_heads=1),
                nn.MaxPool2d(3, 2, 1) if stride > 1 else nn.Identity()
            )
            self.stages.append(stage)
            in_chans = out_dim

        self.fusion = KANFusion(
            feat_dims=[base_dim * 2, base_dim * 3, base_dim * 4, base_dim * 5],
            target_dim=96
        )

        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            # print(x.shape)
        x = self.fusion(features)

        x = self.fc(x)
        return x
