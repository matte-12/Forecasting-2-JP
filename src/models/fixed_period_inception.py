import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock2D(nn.Module):
    """
    Blocco Inception 2D multi-scala.

    Applica più kernel in parallelo alla stessa rappresentazione
    periodica 2D e ne combina i risultati tramite media.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes=(1, 3, 5),
    ):
        super().__init__()

        if not kernel_sizes:
            raise ValueError("kernel_sizes non può essere vuoto")

        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("I kernel devono essere interi positivi e dispari")

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
                for kernel in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ogni ramo osserva il tensore con un campo recettivo diverso.
        branch_outputs = [branch(x) for branch in self.branches]

        # Shape: (num_branches, B, C, cycles, period)
        stacked = torch.stack(branch_outputs, dim=0)

        return stacked.mean(dim=0)


class FixedPeriodInception2D(nn.Module):
    """
    Modello TimesNet-inspired con periodo fissato manualmente.

    Flusso:
        (B, T, C)
        → embedding
        → reshape periodico 2D
        → Inception multi-scala
        → ritorno 1D
        → residual connection
        → proiezione verso pred_len

    Input:
        (batch, seq_len, num_features)

    Output:
        (batch, pred_len, num_features)
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
    ):
        super().__init__()

        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len e pred_len devono essere positivi")

        if period <= 0:
            raise ValueError("period deve essere positivo")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.period = period

        # Numero di cicli necessari per contenere seq_len.
        # ETTh1: 96 / 24 = 4 cicli giornalieri.
        self.num_cycles = math.ceil(seq_len / period)
        self.padded_len = self.num_cycles * period

        # Proiezione delle feature originali nello spazio latente.
        self.embedding = nn.Linear(num_features, d_model)

        # Struttura vicina al TimesBlock:
        # d_model → d_ff → d_model.
        self.inception_1 = InceptionBlock2D(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_sizes=kernel_sizes,
        )

        self.inception_2 = InceptionBlock2D(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_sizes=kernel_sizes,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(d_model)

        # Proiezione temporale seq_len → pred_len.
        self.temporal_projection = nn.Linear(
            seq_len,
            pred_len,
        )

        # Ritorno alle feature originali.
        self.output_projection = nn.Linear(
            d_model,
            num_features,
        )

    def _to_periodic_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trasforma:
            (B, T, d_model)
        in:
            (B, d_model, num_cycles, period)
        """
        batch_size, time_steps, channels = x.shape

        if time_steps != self.seq_len:
            raise ValueError(
                f"seq_len attesa={self.seq_len}, ricevuta={time_steps}"
            )

        padding_length = self.padded_len - time_steps

        if padding_length > 0:
            # Ripete l'ultimo timestep per completare il ciclo.
            padding = x[:, -1:, :].repeat(1, padding_length, 1)
            x = torch.cat([x, padding], dim=1)

        # (B, padded_len, d_model)
        # → (B, cycles, period, d_model)
        x = x.reshape(
            batch_size,
            self.num_cycles,
            self.period,
            channels,
        )

        # Conv2d usa (B, channels, height, width).
        return x.permute(0, 3, 1, 2).contiguous()

    def _to_temporal_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trasforma:
            (B, d_model, num_cycles, period)
        in:
            (B, seq_len, d_model)
        """
        batch_size, channels, _, _ = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()

        x = x.reshape(
            batch_size,
            self.padded_len,
            channels,
        )

        return x[:, :self.seq_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "L'input deve avere shape (batch, seq_len, num_features)"
            )

        if x.shape[2] != self.num_features:
            raise ValueError(
                f"Feature attese={self.num_features}, "
                f"ricevute={x.shape[2]}"
            )

        # (B, T, C) → (B, T, d_model)
        embedded = self.embedding(x)
        residual = embedded

        # (B, T, d_model) → (B, d_model, cycles, period)
        periodic_2d = self._to_periodic_2d(embedded)

        # Estrazione multi-scala intra/inter-periodo.
        features_2d = self.inception_1(periodic_2d)
        features_2d = self.activation(features_2d)
        features_2d = self.dropout(features_2d)
        features_2d = self.inception_2(features_2d)

        # Ritorno allo spazio temporale 1D.
        features_1d = self._to_temporal_1d(features_2d)

        # Residual connection.
        features_1d = self.normalization(
            features_1d + residual
        )

        # La Linear temporale deve ricevere seq_len
        # sull'ultima dimensione.
        features_1d = features_1d.transpose(1, 2)

        # (B, d_model, seq_len) → (B, d_model, pred_len)
        forecast = self.temporal_projection(features_1d)

        # (B, d_model, pred_len) → (B, pred_len, d_model)
        forecast = forecast.transpose(1, 2)

        # (B, pred_len, d_model) → (B, pred_len, num_features)
        return self.output_projection(forecast)


if __name__ == "__main__":
    batch_size = 32
    seq_len = 96
    pred_len = 24
    num_features = 7

    model = FixedPeriodInception2D(
        seq_len=seq_len,
        pred_len=pred_len,
        num_features=num_features,
        period=24,
    )

    dummy_input = torch.randn(
        batch_size,
        seq_len,
        num_features,
    )

    output = model(dummy_input)

    print("Input:", dummy_input.shape)
    print("Output:", output.shape)

    assert output.shape == (
        batch_size,
        pred_len,
        num_features,
    )