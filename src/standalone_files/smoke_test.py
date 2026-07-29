# test per controllare shapes

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import TimeSeriesDataset
from src.models_1d import DLinear, CausalTCN


def run_smoke_test(csv_name: str, batch_size: int = 32, seq_len: int = 96, pred_len: int = 24):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "ETT-small" / csv_name
    dataset = TimeSeriesDataset(
        csv_path=str(csv_path),
        flag="train",
        seq_len=seq_len,
        pred_len=pred_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x, y = next(iter(dataloader))
    print(f"\nDataset: {csv_name}")
    print(f"Input batch  shape: {x.shape}")
    print(f"Target batch shape: {y.shape}")

    assert x.shape == (batch_size, seq_len, 7), f"Input shape sbagliata: {x.shape}"
    assert y.shape == (batch_size, pred_len, 7), f"Target shape sbagliata: {y.shape}"

    dlinear = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=7)
    tcn = CausalTCN(seq_len=seq_len, pred_len=pred_len, enc_in=7)

    with torch.no_grad():
        out_dlinear = dlinear(x)
        out_tcn = tcn(x)

    print(f"DLinear output shape: {out_dlinear.shape}")
    print(f"TCN output shape:     {out_tcn.shape}")

    assert out_dlinear.shape == (batch_size, pred_len, 7), f"DLinear shape sbagliata: {out_dlinear.shape}"
    assert out_tcn.shape == (batch_size, pred_len, 7), f"TCN shape sbagliata: {out_tcn.shape}"


if __name__ == "__main__":
    run_smoke_test("ETTh1.csv")
    run_smoke_test("ETTm1.csv")
    print("\nTutti i test shape sono passati.")