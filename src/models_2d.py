import torch
import torch.nn as nn

class TimesBlock(nn.Module):
    def __init__(self, seq_len, top_k, d_model):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model
        
        # Inception o layer 2D alleggerito
        self.conv2d = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)

    def forward(self, x):
        B, T, C = x.shape
        
        # FFT per estrazione periodi (stub per ablation)
        # fft_out = torch.fft.rfft(x, dim=1)
        # ... logica top_k frequenze ...
        
        # reshape 1D -> 2D (dummy logic per preservare il flusso)
        # forziamo la rete a vedere il tensore come 2D [B, C, p_i, f_i] ?
        # in produzione qui c'è il loop sulle top_k frequenze
        x_2d = x.transpose(1, 2).unsqueeze(-1) # Placeholder shape [B, C, T, 1]
        
        # estrazione feature 2D
        out_2d = self.conv2d(x_2d)
        
        # reshape 2D -> 1D
        out_1d = out_2d.squeeze(-1).transpose(1, 2) # Torna a [B, T, C]
        
        return out_1d

class TimesNet(nn.Module):
    def __init__(self, seq_len=96, pred_len=24, enc_in=7, d_model=32, top_k=3):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # feature embedding, proiex inn spazio latente
        self.embedding = nn.Linear(enc_in, d_model)
        
        # modulo core (ispirato a TimesNet)
        self.times_block = TimesBlock(seq_len, top_k, d_model)
        
        # direct forecasting, proriez finale in orizzonte predittivo
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
        B, T, C = x.shape
        
        # embedding [B, T, C] -> [B, T, d_model]
        enc_out = self.embedding(x)
        
        # feature extraction [B, T, d_model] -> [B, T, d_model]
        enc_out = self.times_block(enc_out)
        
        # proiezione orizzonte: transpose per operare sull'asse del tempo
        # [B, d_model, T]
        enc_out = enc_out.transpose(1, 2) 
        
        # [B, d_model, T] -> [B, d_model, H]
        dec_out = self.projection(enc_out)
        
        # torno a [B, H, d_model]
        dec_out = dec_out.transpose(1, 2)
        
        # proiezione Canali: [B, H, d_model] -> [B, H, C]
        out = self.out_layer(dec_out)

        # altro?
        
        return out

# --- BLOCCO DI TEST INDIPENDENTE --- spostare in file separato?
if __name__ == '__main__':
    # per testare il forward pass prima dell'integrazione
    B, T, C, H = 32, 96, 7, 24
    dummy_input = torch.randn(B, T, C)
    
    model = TimesNet(seq_len=T, pred_len=H, enc_in=C)
    
    try:
        output = model(dummy_input)
        print(f"Forward Pass OK.")
        print(f"Input Shape:  {dummy_input.shape}")
        print(f"Output Shape: {output.shape}")
        
        assert output.shape == (B, H, C), "ERRORE: dim target non allineata."
    except Exception as e:
        print(f"Test Fallito: {e}")