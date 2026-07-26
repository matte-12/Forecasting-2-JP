import math
import torch
import torch.nn as nn

# ============================================================
# 1. BLOCCO MULTI-SCALA INCEPTION
# ============================================================
class MultiScaleInceptionBlock2D(nn.Module):
    """
    Blocco Inception 2D multi-scala.
    Applica kernel 1x1, 3x3 e 5x5 in parallelo e combina i risultati tramite media
    per mantenere l'allineamento matematico con il TimesNet originale.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=(1, 3, 5)):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes non può essere vuoto.")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("I kernel devono essere interi positivi e dispari.")

        self.kernel_sizes = tuple(int(kernel) for kernel in kernel_sizes)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                nn.GELU(),
            ) for kernel in self.kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_outputs, dim=0)
        return stacked.mean(dim=0)

# ============================================================
# 2. BLOCCO DEPTHWISE-SEPARABLE
# ============================================================
class DepthwiseSeparableBlock2D(nn.Module):
    """
    Blocco depthwise-separable per ridurre drasticamente i FLOPs.
    Elabora separatamente ogni canale (groups=in_channels) e fa mixing con la 1x1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================
# 3. BLOCCO GROUP CONVOLUTION
# ============================================================
class GroupConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 4):
        super().__init__()
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("Groups deve essere positivo e divisore esatto dei canali.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================
# 4. BLOCCO CON UN SOLO KERNEL
# ============================================================
class SingleKernelBlock2D(nn.Module):
    """
    Blocco a singola scala. Serve come controllo sperimentale (Baseline Spaziale).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size deve essere un intero positivo e dispari.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================
# 5. LIGHT TIMES BLOCK GENERICO
# ============================================================
class LightTimesBlock(nn.Module):
    """
    Gestisce il reshaping topologico vettorializzato [B, T, C] -> [B, C, F, P]
    e il routing verso la variante architetturale desiderata.
    """
    def __init__(
        self, d_model: int, fixed_period: int = 24, block_type: str = "multiscale",
        kernel_sizes=(1, 3, 5), kernel_size: int = 3, groups: int = 4
    ):
        super().__init__()
        self.fixed_period = max(1, int(fixed_period))

        if block_type == "multiscale":
            self.conv_2d = MultiScaleInceptionBlock2D(d_model, d_model, kernel_sizes)
        elif block_type == "depthwise":
            self.conv_2d = DepthwiseSeparableBlock2D(d_model, d_model)
        elif block_type == "group":
            self.conv_2d = GroupConvBlock2D(d_model, d_model, groups)
        elif block_type == "single_kernel":
            self.conv_2d = SingleKernelBlock2D(d_model, d_model, kernel_size)
        else:
            raise ValueError(f"block_type non riconosciuto: {block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.shape
        period = self.fixed_period
        number_of_cycles = max(1, math.ceil(time_steps / period))
        length_needed = period * number_of_cycles

        # Padding temporale se necessario
        if length_needed > time_steps:
            padding_length = length_needed - time_steps
            padding = x[:, -1:, :].repeat(1, padding_length, 1)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :length_needed, :]

        # [B, T, C] -> [B, C, cicli, periodo]
        x_2d = x_padded.reshape(batch_size, number_of_cycles, period, channels).permute(0, 3, 1, 2).contiguous()
        
        # Convoluzione Spaziale
        out_2d = self.conv_2d(x_2d)

        # [B, C, cicli, periodo] -> [B, T, C]
        out_1d = out_2d.permute(0, 2, 3, 1).reshape(batch_size, length_needed, channels)
        
        # Troncamento e Residual Connection
        return out_1d[:, :time_steps, :] + x

# ============================================================
# 6. MODELLO LIGHT GENERICO
# ============================================================
class LightTimesNet(nn.Module):
    def __init__(
        self, seq_len: int = 96, pred_len: int = 24, enc_in: int = 7, d_model: int = 32,
        fixed_period: int = 24, dropout: float = 0.1, block_type: str = "multiscale",
        kernel_sizes=(1, 3, 5), kernel_size: int = 3, groups: int = 4
    ):
        super().__init__()
        self.embedding = nn.Linear(enc_in, d_model)
        self.times_block = LightTimesBlock(
            d_model, fixed_period, block_type, kernel_sizes, kernel_size, groups
        )
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.Dropout(dropout)
        )
        self.out_layer = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.embedding(x)
        encoded = self.times_block(encoded)
        
        # Proiezione Temporale: [B, T, d_model] -> [B, H, d_model]
        encoded = encoded.transpose(1, 2)
        decoded = self.projection(encoded)
        decoded = decoded.transpose(1, 2)
        
        return self.out_layer(decoded)

# ============================================================
# 7. CLASSI WRAPPER (Per instanziazione diretta da factory)
# ============================================================
class LightTimesNetMultiScale(LightTimesNet):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, fixed_period=24, dropout=0.1, kernel_sizes=(1, 3, 5)):
        super().__init__(seq_len, pred_len, enc_in, d_model, fixed_period, dropout, "multiscale", kernel_sizes)

class LightTimesNetDepthwise(LightTimesNet):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, fixed_period=24, dropout=0.1):
        super().__init__(seq_len, pred_len, enc_in, d_model, fixed_period, dropout, "depthwise")

class LightTimesNetGroup(LightTimesNet):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, fixed_period=24, dropout=0.1, groups=4):
        super().__init__(seq_len, pred_len, enc_in, d_model, fixed_period, dropout, "group", groups=groups)

class LightTimesNetSingleKernel(LightTimesNet):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, fixed_period=24, dropout=0.1, kernel_size=3):
        super().__init__(seq_len, pred_len, enc_in, d_model, fixed_period, dropout, "single_kernel", kernel_size=kernel_size)

# ============================================================
# 8. TEST
# ============================================================
if __name__ == "__main__":
    x = torch.randn(4, 96, 7)
    models = {
        "multiscale": LightTimesNetMultiScale(),
        "depthwise": LightTimesNetDepthwise(),
        "group": LightTimesNetGroup(groups=4),
        "single_kernel": LightTimesNetSingleKernel(kernel_size=3),
    }

    print(f"{'Modello':<14} | Output Shape        | Parametri")
    print("-" * 55)
    for name, model in models.items():
        output = model(x)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<14} | {str(tuple(output.shape)):<19} | {parameters:,}")