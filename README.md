# Studio 2 — Klasifikasi Sampah · Kelompok 15

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |              |     |                  |
| 2  |              |     |                  |
| 3  |              |     |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ResNet-18 pretrained ImageNet (fine-tuning)
- **Strategi pelatihan**: AdamW (lr=3e-4, weight_decay=1e-4), CosineAnnealingLR, CrossEntropyLoss (label smoothing=0.1), 8 epoch, augmentasi (RandomResizedCrop, RandomHorizontalFlip, ColorJitter)
- **Data tambahan**: Tidak ada
- **Ukuran input & preprocessing**: Resize(256), CenterCrop(224), Normalisasi ImageNet

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---|---|
| Preview 1 (14 Okt) | *(baseline)* | 0.9279 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- Eksperimen baseline ResNet-18 pretrained (8 epoch) mencapai akurasi 93.17% dan Macro-F1 0.9279 pada split validasi lokal (my_val).

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
