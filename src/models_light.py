import math
from typing import Type

import torch
import torch.nn as nn


# ============================================================
# 1. BLOCCO MULTI-SCALA INCEPTION
# ============================================================

class MultiScaleInceptionBlock2D(nn.Module):
    """
    Blocco Inception 2D multi-scala.

    Applica più kernel in parallelo alla stessa rappresentazione
    periodica 2D e combina i risultati tramite media.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes=(1, 3, 5),
    ):
        super().__init__()

        if not kernel_sizes:
            raise ValueError(
                "kernel_sizes non può essere vuoto."
            )

        if any(
            kernel <= 0 or kernel % 2 == 0
            for kernel in kernel_sizes
        ):
            raise ValueError(
                "I kernel devono essere interi "
                "positivi e dispari."
            )

        self.kernel_sizes = tuple(
            int(kernel)
            for kernel in kernel_sizes
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.GELU(),
                )
                for kernel in self.kernel_sizes
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        branch_outputs = [
            branch(x)
            for branch in self.branches
        ]

        # [numero_branch, B, C, H, W]
        stacked = torch.stack(
            branch_outputs,
            dim=0,
        )

        # Media tra le scale
        return stacked.mean(dim=0)


# ============================================================
# 2. BLOCCO DEPTHWISE-SEPARABLE
# ============================================================

class DepthwiseSeparableBlock2D(nn.Module):
    """
    Blocco depthwise-separable.

    La depthwise convolution elabora separatamente ogni canale.
    La pointwise 1x1 esegue il mixing tra i canali.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Prima depthwise
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),

            # Prima pointwise
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),

            nn.GELU(),

            # Seconda depthwise
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            ),

            # Seconda pointwise
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),

            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 3. BLOCCO GROUP CONVOLUTION
# ============================================================

class GroupConvBlock2D(nn.Module):
    """
    Blocco basato su group convolution.

    I canali vengono divisi in gruppi. Ogni gruppo viene
    elaborato separatamente da una Conv2D 3x3.

    Una Conv2D 1x1 finale permette il mixing tra gruppi.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 4,
    ):
        super().__init__()

        groups = int(groups)

        if groups <= 0:
            raise ValueError(
                "groups deve essere positivo."
            )

        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels={in_channels} non è "
                f"divisibile per groups={groups}."
            )

        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels={out_channels} non è "
                f"divisibile per groups={groups}."
            )

        self.groups = groups

        self.net = nn.Sequential(
            # Elaborazione separata dei gruppi
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                groups=groups,
            ),
            nn.GELU(),

            # Mixing globale tra i gruppi
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 4. BLOCCO CON UN SOLO KERNEL
# ============================================================

class SingleKernelBlock2D(nn.Module):
    """
    Blocco con una sola scala convoluzionale.

    Il kernel predefinito è 3x3.

    Questa variante serve come controllo sperimentale rispetto
    al blocco multi-scala con kernel 1x1, 3x3 e 5x5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        kernel_size = int(kernel_size)

        if (
            kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError(
                "kernel_size deve essere un intero "
                "positivo e dispari."
            )

        self.kernel_size = kernel_size

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 5. LIGHT TIMES BLOCK GENERICO
# ============================================================

class LightTimesBlock(nn.Module):
    """
    Blocco periodico 2D con backbone selezionabile.

    Trasformazione:

        [B, T, C]
        -> [B, C, numero_cicli, periodo]
        -> blocco 2D
        -> [B, T, C]
    """

    def __init__(
        self,
        d_model: int,
        fixed_period: int = 24,
        block_type: str = "multiscale",
        kernel_sizes=(1, 3, 5),
        kernel_size: int = 3,
        groups: int = 4,
    ):
        super().__init__()

        self.fixed_period = max(
            1,
            int(fixed_period),
        )

        self.block_type = block_type

        if block_type == "multiscale":
            self.conv_2d = (
                MultiScaleInceptionBlock2D(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_sizes=kernel_sizes,
                )
            )

        elif block_type == "depthwise":
            self.conv_2d = (
                DepthwiseSeparableBlock2D(
                    in_channels=d_model,
                    out_channels=d_model,
                )
            )

        elif block_type == "group":
            self.conv_2d = GroupConvBlock2D(
                in_channels=d_model,
                out_channels=d_model,
                groups=groups,
            )

        elif block_type == "single_kernel":
            self.conv_2d = SingleKernelBlock2D(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
            )

        else:
            raise ValueError(
                "block_type non riconosciuto: "
                f"{block_type}. Valori validi: "
                "'multiscale', 'depthwise', "
                "'group', 'single_kernel'."
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            [B, T, C]

        Output:
            [B, T, C]
        """
        if x.ndim != 3:
            raise ValueError(
                "LightTimesBlock richiede input [B,T,C], "
                f"ricevuto {tuple(x.shape)}."
            )

        batch_size, time_steps, channels = (
            x.shape
        )

        period = self.fixed_period

        number_of_cycles = max(
            1,
            math.ceil(time_steps / period),
        )

        length_needed = (
            period * number_of_cycles
        )

        # Padding temporale
        if length_needed > time_steps:
            padding_length = (
                length_needed - time_steps
            )

            padding = x[:, -1:, :].repeat(
                1,
                padding_length,
                1,
            )

            x_padded = torch.cat(
                [x, padding],
                dim=1,
            )

        else:
            x_padded = x[
                :,
                :length_needed,
                :,
            ]

        # [B,T,C] -> [B,cicli,periodo,C]
        x_2d = x_padded.reshape(
            batch_size,
            number_of_cycles,
            period,
            channels,
        )

        # -> [B,C,cicli,periodo]
        x_2d = x_2d.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        # Elaborazione 2D
        out_2d = self.conv_2d(x_2d)

        # [B,C,cicli,periodo]
        # -> [B,cicli,periodo,C]
        out_1d = out_2d.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        # -> [B,length_needed,C]
        out_1d = out_1d.reshape(
            batch_size,
            length_needed,
            channels,
        )

        # Eliminazione del padding
        out_1d = out_1d[
            :,
            :time_steps,
            :,
        ]

        # Residual comune a tutte le varianti
        return out_1d + x


# ============================================================
# 6. MODELLO LIGHT GENERICO
# ============================================================

class LightTimesNet(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        enc_in: int = 7,
        d_model: int = 32,
        fixed_period: int = 24,
        dropout: float = 0.1,
        block_type: str = "multiscale",
        kernel_sizes=(1, 3, 5),
        kernel_size: int = 3,
        groups: int = 4,
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.d_model = int(d_model)

        self.fixed_period = max(
            1,
            int(fixed_period),
        )

        self.block_type = block_type

        self.embedding = nn.Linear(
            in_features=self.enc_in,
            out_features=self.d_model,
        )

        self.times_block = LightTimesBlock(
            d_model=self.d_model,
            fixed_period=self.fixed_period,
            block_type=block_type,
            kernel_sizes=kernel_sizes,
            kernel_size=kernel_size,
            groups=groups,
        )

        self.projection = nn.Sequential(
            nn.Linear(
                in_features=self.seq_len,
                out_features=self.pred_len,
            ),
            nn.Dropout(
                p=float(dropout)
            ),
        )

        self.out_layer = nn.Linear(
            in_features=self.d_model,
            out_features=self.enc_in,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            [B, seq_len, enc_in]

        Output:
            [B, pred_len, enc_in]
        """
        encoded = self.embedding(x)

        encoded = self.times_block(
            encoded
        )

        # [B,T,d_model] -> [B,d_model,T]
        encoded = encoded.transpose(
            1,
            2,
        )

        # seq_len -> pred_len
        decoded = self.projection(
            encoded
        )

        # [B,d_model,H] -> [B,H,d_model]
        decoded = decoded.transpose(
            1,
            2,
        )

        return self.out_layer(decoded)


# ============================================================
# 7. CLASSI SELEZIONABILI CON --model-class
# ============================================================

class LightTimesNetMultiScale(
    LightTimesNet
):
    def __init__(
        self,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=32,
        fixed_period=24,
        dropout=0.1,
        kernel_sizes=(1, 3, 5),
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            fixed_period=fixed_period,
            dropout=dropout,
            block_type="multiscale",
            kernel_sizes=kernel_sizes,
        )


class LightTimesNetDepthwise(
    LightTimesNet
):
    def __init__(
        self,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=32,
        fixed_period=24,
        dropout=0.1,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            fixed_period=fixed_period,
            dropout=dropout,
            block_type="depthwise",
        )


class LightTimesNetGroup(
    LightTimesNet
):
    def __init__(
        self,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=32,
        fixed_period=24,
        dropout=0.1,
        groups=4,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            fixed_period=fixed_period,
            dropout=dropout,
            block_type="group",
            groups=groups,
        )


class LightTimesNetSingleKernel(
    LightTimesNet
):
    def __init__(
        self,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=32,
        fixed_period=24,
        dropout=0.1,
        kernel_size=3,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            fixed_period=fixed_period,
            dropout=dropout,
            block_type="single_kernel",
            kernel_size=kernel_size,
        )


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":
    x = torch.randn(
        4,
        96,
        7,
    )

    models = {
        "multiscale": LightTimesNetMultiScale(),
        "depthwise": LightTimesNetDepthwise(),
        "group": LightTimesNetGroup(
            groups=4
        ),
        "single_kernel": (
            LightTimesNetSingleKernel(
                kernel_size=3
            )
        ),
    }

    for name, model in models.items():
        output = model(x)

        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print(
            f"{name:14s} | "
            f"output={tuple(output.shape)} | "
            f"parameters={parameters:,}"
        )