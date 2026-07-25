import math

import torch
import torch.nn as nn


# ============================================================
# INCEPTION MULTI-SCALA
# ============================================================

class InceptionBlock2D(nn.Module):
    """
    Blocco Inception 2D multi-scala.

    Applica più kernel in parallelo alla stessa rappresentazione
    periodica 2D e combina i risultati tramite media.
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

        kernel_sizes = tuple(
            int(kernel)
            for kernel in kernel_sizes
        )

        if any(
            kernel <= 0 or kernel % 2 == 0
            for kernel in kernel_sizes
        ):
            raise ValueError(
                "I kernel devono essere interi "
                "positivi e dispari."
            )

        self.kernel_sizes = kernel_sizes

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
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

        stacked_outputs = torch.stack(
            branch_outputs,
            dim=0,
        )

        return stacked_outputs.mean(
            dim=0
        )


# ============================================================
# SINGO TIMES BLOCK
# ============================================================

class FixedPeriodTimesBlock(nn.Module):
    """
    Singolo TimesBlock con periodo fissato.

    Input:
        [B, seq_len, d_model]

    Output:
        [B, seq_len, d_model]
    """

    def __init__(
        self,
        seq_len: int,
        period: int,
        d_model: int,
        d_ff: int,
        kernel_sizes=(1, 3, 5),
        dropout: float = 0.1,
    ):
        super().__init__()

        if seq_len <= 0:
            raise ValueError(
                "seq_len deve essere positivo."
            )

        if period <= 0:
            raise ValueError(
                "period deve essere positivo."
            )

        self.seq_len = int(seq_len)
        self.period = int(period)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)

        self.num_cycles = math.ceil(
            self.seq_len / self.period
        )

        self.padded_len = (
            self.num_cycles
            * self.period
        )

        self.inception_1 = InceptionBlock2D(
            in_channels=self.d_model,
            out_channels=self.d_ff,
            kernel_sizes=kernel_sizes,
        )

        self.inception_2 = InceptionBlock2D(
            in_channels=self.d_ff,
            out_channels=self.d_model,
            kernel_sizes=kernel_sizes,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            float(dropout)
        )

        self.normalization = nn.LayerNorm(
            self.d_model
        )

    def _to_periodic_2d(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, T, d_model]
        ->
        [B, d_model, num_cycles, period]
        """
        batch_size, time_steps, channels = (
            x.shape
        )

        if time_steps != self.seq_len:
            raise ValueError(
                f"seq_len attesa={self.seq_len}, "
                f"ricevuta={time_steps}."
            )

        padding_length = (
            self.padded_len
            - time_steps
        )

        if padding_length > 0:
            padding = x[:, -1:, :].repeat(
                1,
                padding_length,
                1,
            )

            x = torch.cat(
                [x, padding],
                dim=1,
            )

        x = x.reshape(
            batch_size,
            self.num_cycles,
            self.period,
            channels,
        )

        return x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    def _to_temporal_1d(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        [B, d_model, num_cycles, period]
        ->
        [B, seq_len, d_model]
        """
        batch_size, channels, _, _ = (
            x.shape
        )

        x = x.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        x = x.reshape(
            batch_size,
            self.padded_len,
            channels,
        )

        return x[
            :,
            :self.seq_len,
            :,
        ]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "FixedPeriodTimesBlock richiede "
                "un input [B, T, d_model]."
            )

        residual = x

        periodic_2d = (
            self._to_periodic_2d(x)
        )

        features_2d = self.inception_1(
            periodic_2d
        )

        features_2d = self.activation(
            features_2d
        )

        features_2d = self.dropout(
            features_2d
        )

        features_2d = self.inception_2(
            features_2d
        )

        features_2d = self.dropout(
            features_2d
        )

        features_1d = (
            self._to_temporal_1d(
                features_2d
            )
        )

        return self.normalization(
            features_1d + residual
        )


# ============================================================
# MODELLO COMPLETO
# ============================================================

class FixedPeriodInception2D(nn.Module):
    """
    Modello TimesNet-inspired con periodo noto.

    num_blocks determina quanti TimesBlock vengono applicati
    consecutivamente.

    Input:
        [B, seq_len, num_features]

    Output:
        [B, pred_len, num_features]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        period: int = 24,
        d_model: int = 32,
        d_ff: int = 64,
        kernel_sizes=(1, 3, 5),
        dropout: float = 0.1,
        num_blocks: int = 1,
    ):
        super().__init__()

        if seq_len <= 0:
            raise ValueError(
                "seq_len deve essere positivo."
            )

        if pred_len <= 0:
            raise ValueError(
                "pred_len deve essere positivo."
            )

        if num_features <= 0:
            raise ValueError(
                "num_features deve essere positivo."
            )

        if period <= 0:
            raise ValueError(
                "period deve essere positivo."
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks deve essere positivo."
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.num_features = int(
            num_features
        )

        self.period = int(period)
        self.fixed_period = int(period)

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.num_blocks = int(
            num_blocks
        )

        self.kernel_sizes = tuple(
            int(kernel)
            for kernel in kernel_sizes
        )

        self.embedding = nn.Linear(
            self.num_features,
            self.d_model,
        )

        self.times_blocks = nn.ModuleList(
            [
                FixedPeriodTimesBlock(
                    seq_len=self.seq_len,
                    period=self.period,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    kernel_sizes=(
                        self.kernel_sizes
                    ),
                    dropout=dropout,
                )
                for _ in range(
                    self.num_blocks
                )
            ]
        )

        self.temporal_projection = nn.Linear(
            self.seq_len,
            self.pred_len,
        )

        self.output_projection = nn.Linear(
            self.d_model,
            self.num_features,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "L'input deve avere shape "
                "[B, seq_len, num_features]."
            )

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"seq_len attesa={self.seq_len}, "
                f"ricevuta={x.shape[1]}."
            )

        if x.shape[2] != self.num_features:
            raise ValueError(
                f"Feature attese={self.num_features}, "
                f"ricevute={x.shape[2]}."
            )

        # [B, T, C] -> [B, T, d_model]
        features = self.embedding(x)

        # Ripete il TimesBlock num_blocks volte.
        for times_block in self.times_blocks:
            features = times_block(
                features
            )

        # [B, T, d_model]
        # -> [B, d_model, T]
        features = features.transpose(
            1,
            2,
        )

        # [B, d_model, T]
        # -> [B, d_model, H]
        forecast = self.temporal_projection(
            features
        )

        # [B, d_model, H]
        # -> [B, H, d_model]
        forecast = forecast.transpose(
            1,
            2,
        )

        # [B, H, d_model]
        # -> [B, H, num_features]
        return self.output_projection(
            forecast
        )


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    batch_size = 32
    seq_len = 96
    pred_len = 24
    num_features = 7

    dummy_input = torch.randn(
        batch_size,
        seq_len,
        num_features,
    )

    for num_blocks in (1, 2, 3):
        model = FixedPeriodInception2D(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            period=24,
            d_model=32,
            d_ff=64,
            kernel_sizes=(1, 3, 5),
            dropout=0.1,
            num_blocks=num_blocks,
        )

        output = model(
            dummy_input
        )

        parameter_count = sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )

        print(
            f"num_blocks={num_blocks} | "
            f"input={tuple(dummy_input.shape)} | "
            f"output={tuple(output.shape)} | "
            f"parameters={parameter_count:,}"
        )

        assert output.shape == (
            batch_size,
            pred_len,
            num_features,
        )