import math
import torch
import torch.nn as nn

class SingleKernelBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class MultiKernelInception2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 3 rami con campi recettivi diversi (1x1, 3x3, 5x5) per catturare 
        # dipendenze spaziali a scale differenti senza ridondanza matematica.
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.activation = nn.GELU()
        
        # Pointwise per fondere i rami
        self.mixer = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1) # [B, 3*C, F, P]
        out = self.activation(self.mixer(out))     # [B, C, F, P]
        return out

class LightTimesBlock(nn.Module):
    def __init__(self, d_model, fixed_period=24, kernel_type="multi"):
        super().__init__()
        self.fixed_period = max(1, int(fixed_period))
        
        if kernel_type == "single":
            self.conv_2d = SingleKernelBlock2D(channels=d_model)
        elif kernel_type == "multi":
            self.conv_2d = MultiKernelInception2D(channels=d_model)
        else:
            raise ValueError("kernel_type deve essere 'single' o 'multi'")

    def forward(self, x):
        batch_size, time_steps, channels = x.shape
        period = self.fixed_period
        freq = max(1, math.ceil(time_steps / period))
        length_needed = period * freq

        # padding temporale (se la finestra non è un multiplo perfetto del periodo)
        if length_needed > time_steps:
            pad_len = length_needed - time_steps
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x[:, :length_needed, :]

        # [B, freq, period, C] -> [B, C, freq, period]
        x_2d = x_padded.reshape(batch_size, freq, period, channels).permute(0, 3, 1, 2).contiguous()
        
        out_2d = self.conv_2d(x_2d)
        
        out_1d = out_2d.permute(0, 2, 3, 1).reshape(batch_size, length_needed, channels)
        return out_1d[:, :time_steps, :] + x

class LightTimesNet(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, fixed_period=24, kernel_type="multi"):
        super().__init__()
        self.embedding = nn.Linear(enc_in, d_model)
        self.times_block = LightTimesBlock(d_model=d_model, fixed_period=fixed_period, kernel_type=kernel_type)
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