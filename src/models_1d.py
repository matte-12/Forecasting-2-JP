import torch
import torch.nn as nn

# based on our researches DLinear and TCN-Based networks are those performing best among many 1D possibilities

# ==========================================
# DLINEAR (MLP-Based)
# ==========================================
class MovingAvg(nn.Module):
    """
    Blocco di media mobile per estrarre la componente di trend.
    Utilizza padding per mantenere invariata la dimensione temporale.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding front/end per pareggiare il kernel
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x

class SeriesDecomp(nn.Module):
    """
    Scomposizione del segnale in componente stagionale (residuale) e trend (low-pass).
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear: Channel-independent Decomposition Linear Model.
    Approccio diretto che sfrutta operazioni lineari su asse temporale.
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.decomp = SeriesDecomp(kernel_size=25)
        # Mapping lineare T -> H. Pesi condivisi per tutti i canali (Channel Independence)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        seasonal_init, trend_init = self.decomp(x)
        
        # PyTorch nn.Linear agisce sull'ultima dimensione, quindi mappa T su H
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        
        x_out = seasonal_output + trend_output # [B, C, H]
        
        # [B, C, H] -> [B, H, C]
        return x_out.transpose(1, 2)


# ==========================================
# TEMPORAL CONVOLUTIONAL NETWORK (TCN-Based)
# ==========================================
class Chomp1d(nn.Module):
    """
    Rimuove il padding asimmetrico finale per garantire la rigorosa causalità della convoluzione.
    Evita la dipendenza da valori futuri all'interno della medesima finestra.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Connessione residuale 1x1 se i canali variano
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class CausalTCN(nn.Module):
    """
    Causal TCN per Direct Forecasting.
    Usa dilatazioni esponenziali per massimizzare il campo recettivo senza esplosione di parametri.
    """
    def __init__(self, seq_len, pred_len, enc_in, num_channels=[32, 64], kernel_size=3, dropout=0.2):
        super(CausalTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = enc_in if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.tcn = nn.Sequential(*layers)
        
        # Proiezione canali latenti ai canali reali (enc_in)
        self.channel_proj = nn.Conv1d(num_channels[-1], enc_in, kernel_size=1)
        # Direct forecasting: proiezione T -> H
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        # Estrazione feature convoluzionali 1D
        x = self.tcn(x)                  # [B, hidden_channels, T]
        x = self.channel_proj(x)         # [B, C, T]
        
        # Mapping temporale
        x = self.temporal_proj(x)        # [B, C, H]
        
        # [B, C, H] -> [B, H, C]
        return x.transpose(1, 2)

# ==========================================
# SANITY CHECK I/O
# ==========================================
if __name__ == '__main__':
    B, T, C, H = 32, 96, 7, 24
    dummy_input = torch.randn(B, T, C)
    
    # Test DLinear
    model_dlinear = DLinear(seq_len=T, pred_len=H, enc_in=C)
    out_dlinear = model_dlinear(dummy_input)
    assert out_dlinear.shape == (B, H, C), "Errore shape in DLinear"
    
    # Test TCN
    model_tcn = CausalTCN(seq_len=T, pred_len=H, enc_in=C)
    out_tcn = model_tcn(dummy_input)
    assert out_tcn.shape == (B, H, C), "Errore shape in CausalTCN"
    
    print("Modelli 1D pronti: Le architetture rispettano le firme tensoriali.")