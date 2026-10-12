"""Script Pelatihan Model — Studio 2 Klasifikasi Sampah (Kelompok 15).

Contoh penggunaan:
    python train.py --data ./train --epochs 8
    python train.py --data ./train --epochs 10 --lr 1e-4 --out checkpoints/exp1.pth
    python train.py --data ./train --epochs 10 --export-submission  # Simpan langsung ke submission/model.pth jika terbaik
"""
import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from submission.model import get_model, get_transform


def make_local_split(src, dst=".", frac=0.15, seed=42):
    """Membuat split lokal train (85%) dan val (15%) sekali saja."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va

    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def main():
    parser = argparse.ArgumentParser(description="Training model klasifikasi sampah")
    parser.add_argument("--data", default="./train", help="Folder dataset train asli")
    parser.add_argument("--epochs", type=int, default=8, help="Jumlah epoch pelatihan")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="checkpoints/latest_model.pth", help="Path penyimpanan bobot checkpoint")
    parser.add_argument("--export-submission", action="store_true", help="Salin model terbaik langsung ke submission/model.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device yang digunakan: {device}")

    # Siapkan split dataset lokal jika belum ada
    tr_dir, va_dir = make_local_split(args.data, ".", frac=0.15, seed=args.seed)

    # Augmentasi data training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_ds = datasets.ImageFolder(tr_dir, transform=train_tf)
    va_ds = datasets.ImageFolder(va_dir, transform=get_transform())
    print(f"Kelas: {tr_ds.classes} | Data Train: {len(tr_ds)} | Data Val: {len(va_ds)}")

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Inisialisasi model
    model = get_model()
    # Muat pretrained ImageNet untuk backbone ResNet-18
    pre = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
    pre = {k: v for k, v in pre.items() if not k.startswith("fc.")}
    model.load_state_dict(pre, strict=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Pastikan folder penyimpanan tersedia
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        scheduler.step()

        # Evaluasi validasi
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()

        val_acc = correct / len(va_ds)
        avg_loss = total_loss / len(tr_ds)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(out_path))
            print(f"  -> Model terbaik baru disimpan ke {out_path} (Val Acc: {best_acc:.4f})")
            if args.export_submission:
                torch.save(model.state_dict(), "submission/model.pth")
                print("  -> Model berhasil diekspor ke submission/model.pth")

    print(f"\nPelatihan selesai! Val Acc terbaik: {best_acc:.4f}")
    print(f"Untuk menguji metrik Macro-F1 resmi, jalankan:")
    print(f"python evaluate.py --submission ./submission --data ./my_val")


if __name__ == "__main__":
    main()
