import math
from typing import Sequence

import torch
import torch.nn as nn


# ============================================================
# FFT: ESTRAZIONE DEI PERIODI DOMINANTI
# ============================================================

def fft_for_period(
    x: torch.Tensor,
    top_k: int,
) -> tuple[list[int], torch.Tensor]:
    """
    Estrae i periodi dominanti tramite FFT.

    Args:
        x:
            Tensore [B, T, C].

        top_k:
            Numero di frequenze/periodi da selezionare.

    Returns:
        periods:
            Lista di periodi interi.

        period_weights:
            Ampiezze FFT per ogni campione e periodo,
            shape [B, effective_top_k].
    """
    if x.ndim != 3:
        raise ValueError(
            "fft_for_period richiede un input [B, T, C]."
        )

    batch_size, time_steps, _ = x.shape

    if time_steps < 2:
        raise ValueError(
            "La sequenza deve contenere almeno 2 timestep."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k deve essere positivo."
        )

    # FFT reale lungo l'asse temporale.
    frequency_spectrum = torch.fft.rfft(
        x,
        dim=1,
    )

    amplitude = frequency_spectrum.abs()

    # Media su batch e canali:
    # [B, F, C] -> [F]
    global_amplitude = amplitude.mean(
        dim=(0, 2)
    )

    # Escludiamo la componente continua a frequenza zero.
    global_amplitude = global_amplitude.clone()

    if global_amplitude.numel() > 0:
        global_amplitude[0] = 0.0

    maximum_available_frequencies = max(
        1,
        global_amplitude.numel() - 1,
    )

    effective_top_k = min(
        int(top_k),
        maximum_available_frequencies,
    )

    _, frequency_indices = torch.topk(
        global_amplitude,
        k=effective_top_k,
    )

    frequency_indices = frequency_indices.clamp(
        min=1
    )

    # Conversione frequenza -> periodo.
    periods = [
        max(
            1,
            int(
                math.ceil(
                    time_steps
                    / int(frequency_index.item())
                )
            ),
        )
        for frequency_index in frequency_indices
    ]

    # Ampiezza per campione:
    # [B, F, C] -> [B, F]
    sample_amplitude = amplitude.mean(
        dim=2
    )

    period_weights = sample_amplitude[
        :,
        frequency_indices,
    ]

    if period_weights.shape != (
        batch_size,
        effective_top_k,
    ):
        raise RuntimeError(
            "Shape dei pesi FFT non valida: "
            f"{tuple(period_weights.shape)}."
        )

    return periods, period_weights


# ============================================================
# INCEPTION MULTI-SCALA 1, 3, 5
# ============================================================

class InceptionBlock2D(nn.Module):
    """
    InceptionBlock 2D con kernel multi-scala.

    Ogni blocco applica in parallelo:

        1x1
        3x3
        5x5

    Le uscite vengono combinate tramite media.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()

        kernel_sizes = tuple(
            int(kernel)
            for kernel in kernel_sizes
        )

        if not kernel_sizes:
            raise ValueError(
                "kernel_sizes non può essere vuoto."
            )

        if any(
            kernel <= 0 or kernel % 2 == 0
            for kernel in kernel_sizes
        ):
            raise ValueError(
                "I kernel devono essere positivi e dispari."
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
# SINGO TIMESBLOCK
# ============================================================

class TimesBlock(nn.Module):
    """
    TimesBlock con:

        FFT
        -> top-k periodi
        -> reshape 2D per ogni periodo
        -> Inception 1,3,5
        -> aggregazione pesata
        -> residual connection

    Ogni TimesBlock contiene due InceptionBlock consecutivi:

        d_model -> d_ff
        d_ff -> d_model
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        top_k: int = 3,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        dropout: float = 0.1,
        use_fft: bool = True,
        fixed_period: int = 24,
    ):
        super().__init__()

        if seq_len <= 0:
            raise ValueError(
                "seq_len deve essere positivo."
            )

        if d_model <= 0 or d_ff <= 0:
            raise ValueError(
                "d_model e d_ff devono essere positivi."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k deve essere positivo."
            )

        if fixed_period <= 0:
            raise ValueError(
                "fixed_period deve essere positivo."
            )

        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.top_k = int(top_k)

        self.use_fft = bool(use_fft)
        self.fixed_period = int(
            fixed_period
        )

        self.kernel_sizes = tuple(
            int(kernel)
            for kernel in kernel_sizes
        )

        self.inception_1 = InceptionBlock2D(
            in_channels=self.d_model,
            out_channels=self.d_ff,
            kernel_sizes=self.kernel_sizes,
        )

        self.inception_2 = InceptionBlock2D(
            in_channels=self.d_ff,
            out_channels=self.d_model,
            kernel_sizes=self.kernel_sizes,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            float(dropout)
        )

        self.normalization = nn.LayerNorm(
            self.d_model
        )

    def _periodic_processing(
        self,
        x: torch.Tensor,
        period: int,
    ) -> torch.Tensor:
        """
        Trasforma e processa un singolo periodo.

        Input:
            [B, T, d_model]

        Output:
            [B, T, d_model]
        """
        batch_size, time_steps, channels = x.shape

        period = max(
            1,
            int(period),
        )

        number_of_cycles = math.ceil(
            time_steps / period
        )

        padded_length = (
            number_of_cycles
            * period
        )

        padding_length = (
            padded_length
            - time_steps
        )

        if padding_length > 0:
            padding = x[
                :,
                -1:,
                :,
            ].repeat(
                1,
                padding_length,
                1,
            )

            x_padded = torch.cat(
                [x, padding],
                dim=1,
            )

        else:
            x_padded = x

        # [B, padded_T, C]
        # -> [B, cycles, period, C]
        x_2d = x_padded.reshape(
            batch_size,
            number_of_cycles,
            period,
            channels,
        )

        # -> [B, C, cycles, period]
        x_2d = x_2d.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        # Primo InceptionBlock multi-scala.
        x_2d = self.inception_1(
            x_2d
        )

        x_2d = self.activation(
            x_2d
        )

        x_2d = self.dropout(
            x_2d
        )

        # Secondo InceptionBlock multi-scala.
        x_2d = self.inception_2(
            x_2d
        )

        x_2d = self.dropout(
            x_2d
        )

        # [B, C, cycles, period]
        # -> [B, cycles, period, C]
        x_1d = x_2d.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        # -> [B, padded_T, C]
        x_1d = x_1d.reshape(
            batch_size,
            padded_length,
            channels,
        )

        # Rimuove il padding.
        return x_1d[
            :,
            :time_steps,
            :,
        ]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TimesBlock richiede un input [B, T, C]."
            )

        residual = x

        if self.use_fft:
            periods, period_weights = fft_for_period(
                x=x,
                top_k=self.top_k,
            )

        else:
            periods = [
                self.fixed_period
            ]

            period_weights = torch.ones(
                x.shape[0],
                1,
                device=x.device,
                dtype=x.dtype,
            )

        period_outputs = []

        for period in periods:
            period_output = (
                self._periodic_processing(
                    x=x,
                    period=period,
                )
            )

            period_outputs.append(
                period_output
            )

        # [B, T, C, K]
        stacked_outputs = torch.stack(
            period_outputs,
            dim=-1,
        )

        # Normalizzazione dei pesi dei periodi.
        normalized_weights = torch.softmax(
            period_weights,
            dim=1,
        )

        # [B, K] -> [B, 1, 1, K]
        normalized_weights = (
            normalized_weights[
                :,
                None,
                None,
                :,
            ]
        )

        aggregated_output = (
            stacked_outputs
            * normalized_weights
        ).sum(
            dim=-1
        )

        return self.normalization(
            aggregated_output
            + residual
        )


# ============================================================
# TIMESNET ORIGINALE MODIFICATO
# ============================================================

class TimesNetOriginal(nn.Module):
    """
    TimesNet con:

        - FFT attivabile;
        - top_k variabile;
        - kernel multi-scala fissi 1,3,5;
        - numero di TimesBlock variabile;
        - due InceptionBlock per ogni TimesBlock.

    Input:
        [B, seq_len, enc_in]

    Output:
        [B, pred_len, enc_in]
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        enc_in: int = 7,
        d_model: int = 32,
        d_ff: int = 64,
        top_k: int = 3,
        num_blocks: int = 2,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        dropout: float = 0.1,
        use_fft: bool = True,
        fixed_period: int = 24,
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

        if enc_in <= 0:
            raise ValueError(
                "enc_in deve essere positivo."
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks deve essere positivo."
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)

        self.top_k = int(top_k)
        self.num_blocks = int(
            num_blocks
        )

        self.use_fft = bool(
            use_fft
        )

        self.fixed_period = int(
            fixed_period
        )

        self.kernel_sizes = tuple(
            int(kernel)
            for kernel in kernel_sizes
        )

        self.embedding = nn.Linear(
            self.enc_in,
            self.d_model,
        )

        self.times_blocks = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=self.seq_len,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    top_k=self.top_k,
                    kernel_sizes=self.kernel_sizes,
                    dropout=dropout,
                    use_fft=self.use_fft,
                    fixed_period=self.fixed_period,
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

        self.dropout = nn.Dropout(
            float(dropout)
        )

        self.output_projection = nn.Linear(
            self.d_model,
            self.enc_in,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "L'input deve avere shape "
                "[B, seq_len, enc_in]."
            )

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"seq_len attesa={self.seq_len}, "
                f"ricevuta={x.shape[1]}."
            )

        if x.shape[2] != self.enc_in:
            raise ValueError(
                f"Feature attese={self.enc_in}, "
                f"ricevute={x.shape[2]}."
            )

        # [B, T, enc_in]
        # -> [B, T, d_model]
        features = self.embedding(
            x
        )

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

        forecast = self.dropout(
            forecast
        )

        # [B, d_model, H]
        # -> [B, H, d_model]
        forecast = forecast.transpose(
            1,
            2,
        )

        # [B, H, d_model]
        # -> [B, H, enc_in]
        return self.output_projection(
            forecast
        )


# Alias opzionale.
TimesNet = TimesNetOriginal


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    batch_size = 4
    seq_len = 96
    pred_len = 24
    enc_in = 7

    dummy_input = torch.randn(
        batch_size,
        seq_len,
        enc_in,
    )

    for top_k in (1, 2, 3):
        for num_blocks in (1, 2, 3):
            model = TimesNetOriginal(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                d_model=32,
                d_ff=64,
                top_k=top_k,
                num_blocks=num_blocks,
                kernel_sizes=(1, 3, 5),
                dropout=0.1,
                use_fft=True,
                fixed_period=24,
            )

            output = model(
                dummy_input
            )

            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )

            print(
                f"top_k={top_k} | "
                f"num_blocks={num_blocks} | "
                f"output={tuple(output.shape)} | "
                f"parameters={parameter_count:,}"
            )

            assert output.shape == (
                batch_size,
                pred_len,
                enc_in,
            )