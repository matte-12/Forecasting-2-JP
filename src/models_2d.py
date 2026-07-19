import math
import torch
import torch.nn as nn


class InceptionBlock2D(nn.Module):
    """
    Blocco 2D leggero per catturare pattern intra-periodo e inter-periodo.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TimesBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        top_k,
        d_model,
        use_fft=True,
        fixed_period=24,
        use_inception=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model
        self.use_fft = use_fft
        self.fixed_period = max(1, int(fixed_period))
        self.use_inception = use_inception

        # blocco che processerà i tensori reshaped
        if self.use_inception:
            self.conv_2d = InceptionBlock2D(
                in_channels=d_model,
                out_channels=d_model,
            )
        else:
            self.conv_2d = nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
            )

    @staticmethod
    def _pad_time_single(x_1d, target_length):
        """
        x_1d: [T, C]
        """
        time_steps, _ = x_1d.shape
        if target_length <= time_steps:
            return x_1d[:target_length, :]

        pad_len = target_length - time_steps
        pad = x_1d[-1:, :].repeat(pad_len, 1)
        return torch.cat([x_1d, pad], dim=0)

    @staticmethod
    def _pad_time_batch(x, target_length):
        """
        x: [B, T, C]
        """
        batch_size, time_steps, _ = x.shape
        if target_length <= time_steps:
            return x[:, :target_length, :]

        pad_len = target_length - time_steps
        pad = x[:, -1:, :].repeat(1, pad_len, 1)
        return torch.cat([x, pad], dim=1)

    def _reshape_2d_single(self, x_1d, period, freq):
        """
        x_1d: [T, C]
        return: [T, C]
        """
        time_steps, channels = x_1d.shape
        length_needed = period * freq

        x_1d = self._pad_time_single(x_1d, length_needed)

        x_2d = x_1d.reshape(period, freq, channels).permute(2, 0, 1).unsqueeze(0)
        out_2d = self.conv_2d(x_2d)
        out_1d = out_2d.squeeze(0).permute(1, 2, 0).reshape(length_needed, channels)

        return out_1d[:time_steps, :]

    def _reshape_2d_batch(self, x, period, freq):
        """
        x: [B, T, C]
        return: [B, T, C]
        """
        batch_size, time_steps, channels = x.shape
        length_needed = period * freq

        x = self._pad_time_batch(x, length_needed)

        x_2d = x.reshape(batch_size, period, freq, channels).permute(0, 3, 1, 2).contiguous()
        out_2d = self.conv_2d(x_2d)
        out_1d = out_2d.permute(0, 2, 3, 1).reshape(batch_size, length_needed, channels)

        return out_1d[:, :time_steps, :]
    
    def forward(self, x):
        """
        Input:  [B, T, C]
        Output: [B, T, C]
        """
        batch_size, time_steps, channels = x.shape

        if not self.use_fft:
            period = self.fixed_period
            freq = max(1, math.ceil(time_steps / period))
            return self._reshape_2d_batch(x, period, freq) + x

        # fft per calcolo ampiezze, rfft lungo asse temporale
        xf = torch.fft.rfft(x, dim=1)
        amplitudes = xf.abs().mean(dim=2)

        # ampiezza media sui canali per trovare periodi globali
        # frequenza 0 sulla componentecontinua --> no bias
        amplitudes = amplitudes.clone()
        amplitudes[:, 0] = 0

        candidate_count = amplitudes.shape[1] - 1
        if candidate_count <= 0:
            return x

        effective_top_k = min(self.top_k, amplitudes.shape[1] - 1)
        top_indices = torch.topk(amplitudes[:, 1:], k=effective_top_k, dim=1).indices + 1

        outputs = []

        # iterativo da evitare per full batch, ma qui con B=32 impatto trascurabile ora
        # poi in ablation togliamo e mettiamo [B, C, p, f] per avere padding e reshaping in un solo colpo dentro al tensore'?
        for i in range(batch_size):
            batch_outputs = []
            batch_weights = torch.softmax(amplitudes[i, top_indices[i]], dim=0)

            for j in range(effective_top_k):
                freq_idx = int(top_indices[i, j].item())
                freq_idx = max(freq_idx, 1)

                period = max(1, math.ceil(time_steps / freq_idx))
                out_1d = self._reshape_2d_single(x[i], period, freq_idx)
                batch_outputs.append(out_1d)

            batch_outputs = torch.stack(batch_outputs, dim=0)
            fused_out = torch.sum(
                batch_outputs * batch_weights.view(-1, 1, 1),
                dim=0,
            )
            outputs.append(fused_out)

        final_out = torch.stack(outputs, dim=0)
        return final_out + x


class TimesNet(nn.Module):
    def __init__(
        self,
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=32,
        top_k=3,
        use_fft=True,
        fixed_period=24,
        use_inception=True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embedding = nn.Linear(enc_in, d_model)
        self.times_block = TimesBlock(
            seq_len=seq_len,
            top_k=top_k,
            d_model=d_model,
            use_fft=use_fft,
            fixed_period=fixed_period,
            use_inception=use_inception,
        )

        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.Dropout(0.1)
        )

        self.out_layer = nn.Linear(d_model, enc_in)

    def forward(self, x):
        """
        Input:  [B, T, C]
        Output: [B, H, C]
        """
        enc_out = self.embedding(x)
        enc_out = self.times_block(enc_out)
        enc_out = enc_out.transpose(1, 2)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.transpose(1, 2)
        out = self.out_layer(dec_out)
        return out


if __name__ == "__main__":
    B, T, C, H = 32, 96, 7, 24
    dummy_input = torch.randn(B, T, C)

    model = TimesNet(
        seq_len=T,
        pred_len=H,
        enc_in=C,
        use_fft=True,
        use_inception=True,
        fixed_period=24,
    )
    output = model(dummy_input)

    print("Forward Pass OK.")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")

    assert output.shape == (B, H, C), "ERRORE: dim target non allineata."