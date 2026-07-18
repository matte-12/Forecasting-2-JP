import math
import torch
import torch.nn as nn


class InceptionBlock2D(nn.Module):
    """
    blocco 2D lightweight per catturare pattern intra-periodo e inter-periodo,
    per ora ha due conv 2D standard, così resta stabile e facile da ablationare,
    singola convoluz per ora, poi in ablation rivediamo
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
    def __init__(self, seq_len, top_k, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model

        # blocco che processerà i tensori reshaped
        self.conv_2d = InceptionBlock2D(in_channels=d_model, out_channels=d_model)

    def forward(self, x):
        """
        Input:  [B, T, C]
        Output: [B, T, C]
        """
        batch_size, time_steps, channels = x.shape

        # fft per calcolo ampiezze, rfft lungo asse temporale
        xf = torch.fft.rfft(x, dim=1)
        amplitudes = xf.abs().mean(dim=2)

        # ampiezza media sui canali per trovare periodi globali
        # frequenza 0 sulla componentecontinua --> no bias
        amplitudes = amplitudes.clone()
        amplitudes[:, 0] = 0

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
                freq_idx = max(freq_idx, 1) # evita /0

                period = max(1, math.ceil(time_steps / freq_idx)) #per evitare troncamenti
                length_needed = period * freq_idx
                pad_len = length_needed - time_steps

                if pad_len > 0:
                    padded_x = torch.cat([x[i], x[i, -pad_len:, :]], dim=0)
                else:
                    padded_x = x[i, :length_needed, :]

                x_2d = padded_x.reshape(period, freq_idx, channels).permute(2, 0, 1).unsqueeze(0)
                out_2d = self.conv_2d(x_2d)
                out_1d = out_2d.squeeze(0).permute(1, 2, 0).reshape(length_needed, channels)
                out_1d = out_1d[:time_steps, :]

                batch_outputs.append(out_1d)

            batch_outputs = torch.stack(batch_outputs, dim=0)
            fused_out = torch.sum(batch_outputs * batch_weights.view(-1, 1, 1), dim=0)
            outputs.append(fused_out)

        final_out = torch.stack(outputs, dim=0)
        return final_out + x


class TimesNet(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embedding = nn.Linear(enc_in, d_model)
        self.times_block = TimesBlock(seq_len, top_k, d_model)

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

    model = TimesNet(seq_len=T, pred_len=H, enc_in=C)
    output = model(dummy_input)

    print("Forward Pass OK.")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")

    assert output.shape == (B, H, C), "ERRORE: dim target non allineata."