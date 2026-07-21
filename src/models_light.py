import math
import torch
import torch.nn as nn

class InceptionBlock2D(nn.Module):
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

class LightTimesBlock(nn.Module):
    """
    Modulo 2D vettorializzato a periodo fisso. Sostituisce la scoperta
    stocastica della FFT con l'iniezione di conoscenza a priori (Domain Knowledge).
    ANdiamo a sfruttare i periodi temporali già noti invece del doppio for che faceva i reshape
    """
    def __init__(self, d_model, fixed_period=24):
        super().__init__()
        self.fixed_period = max(1, int(fixed_period))
        self.conv_2d = InceptionBlock2D(in_channels=d_model, out_channels=d_model)

    def forward(self, x):
        """
        Input:  [B, T, C]
        Output: [B, T, C]
        """
        batch_size, time_steps, channels = x.shape
        period = self.fixed_period
        freq = max(1, math.ceil(time_steps / period))
        length_needed = period * freq

        # 1. Padding temporale (se la finestra non è un multiplo perfetto del periodo)
        if length_needed > time_steps:
            pad_len = length_needed - time_steps
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x[:, :length_needed, :]

        # 2. Reshape Topologico: [B, freq, period, C]
        # Cruciale: 'freq' sono le righe (giorni), 'period' le colonne (ore)
        x_2d = x_padded.reshape(batch_size, freq, period, channels)

        # 3. Permute per Conv2D: [B, C, H, W] -> [B, C, freq, period]
        x_2d = x_2d.permute(0, 3, 1, 2).contiguous()

        # 4. Convoluzione
        out_2d = self.conv_2d(x_2d)

        # 5. Ritorno allo spazio 1D e troncamento
        out_1d = out_2d.permute(0, 2, 3, 1).reshape(batch_size, length_needed, channels)
        out_1d = out_1d[:, :time_steps, :]

        # Residual connection
        return out_1d + x

class LightTimesNet(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, fixed_period=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embedding = nn.Linear(enc_in, d_model)
        self.times_block = LightTimesBlock(d_model=d_model, fixed_period=fixed_period)
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.Dropout(0.1)
        )
        self.out_layer = nn.Linear(d_model, enc_in)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out = self.times_block(enc_out)
        
        # Mapping Temporale
        enc_out = enc_out.transpose(1, 2)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.transpose(1, 2)
        
        return self.out_layer(dec_out)