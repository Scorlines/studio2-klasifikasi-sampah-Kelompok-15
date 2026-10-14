# Studio 2 — Klasifikasi Sampah · Kelompok 15

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Fadhel             | 123140106    |                  |
| 2  | Abel Fortino             | 123140111     |                  |
| 3  | Gilang Surya Agung              | 123140187     |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ResNet-18 pretrained ImageNet (fine-tuning)
- **Strategi pelatihan**: AdamW (lr=3e-4, weight_decay=1e-4), CosineAnnealingLR, CrossEntropyLoss (label smoothing=0.1), 8 epoch, batch size 16, seed 42
- **Augmentasi training**: RandomResizedCrop(224, scale=(0.6, 1.0)), RandomHorizontalFlip, ColorJitter (brightness/contrast/saturation=0.2)
- **Split lokal**: 2.753 gambar JPG dibagi per kelas menjadi 2.343 gambar training dan 410 gambar validasi (sekitar 85%/15%)
- **Pemilihan checkpoint**: akurasi validasi tertinggi; pada percobaan ini diperoleh pada epoch 8
- **Data tambahan**: Tidak ada
- **Ukuran input & preprocessing**: Resize(256), CenterCrop(224), Normalisasi ImageNet

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---|---|
| Preview 1 (14 Okt) | percobaan 1 | **0.9330** | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Hasil `percobaan1` pada 410 gambar validasi lokal (`my_val`): akurasi **93.66%** dan Macro-F1 **0.9330**. Skor resmi Preview 1 pada validation set tersembunyi menunggu evaluasi asisten.

| Kelas | F1 lokal |
|---|---:|
| botol_plastik | 0.9677 |
| gelas_plastik | 0.9150 |
| kertas_kardus | 0.8952 |
| saset_kemasan | 0.9458 |
| styrofoam | 0.9412 |

Confusion matrix (baris = label asli, kolom = prediksi; urutan kelas sesuai tabel di atas):

```text
                 botol  gelas  kertas  saset  styrofoam
botol_plastik       75      1       0      2         0
gelas_plastik        1     70       2      1         4
kertas_kardus        0      3      47      2         2
saset_kemasan        1      0       1     96         2
styrofoam           0      1       1      2        96
```

Baseline sebelumnya tercatat memiliki akurasi 93.17% dan Macro-F1 0.9279. Pada percobaan terbaru, kelas `kertas_kardus` masih memiliki F1 terendah (0.8952).

## Cara Reproduksi

Jalankan perintah berikut dari folder repositori menggunakan PowerShell di Windows dengan GPU NVIDIA.

### 1. Siapkan environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Pastikan pengecekan menghasilkan `CUDA: True` untuk menggunakan GPU. Jika environment sudah siap, lanjutkan ke langkah berikutnya.

### 2. Siapkan dataset

Unduh dataset dari tautan pada [halaman resmi Studio 2](https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/#dataset), lalu ekstrak folder `train` di sebelah folder repositori:

```text
Tugas2/
├── train/
│   ├── botol_plastik/
│   ├── gelas_plastik/
│   ├── kertas_kardus/
│   ├── saset_kemasan/
│   └── styrofoam/
└── studio2-klasifikasi-sampah-Kelompok-15/
    └── train.py
```

Skrip pembagian data membaca file `*.jpg` langsung di dalam setiap folder kelas. Dataset tidak diunggah ke repositori publik, sesuai aturan tugas.

### 3. Latih model

```powershell
.\.venv\Scripts\python.exe -u train.py --data "..\train" --epochs 8 --batch-size 16 --lr 3e-4 --seed 42 --out "checkpoints\percobaan1.pth"
```

Sesuaikan `--data` jika dataset berada di lokasi lain. Skrip membuat `my_train` dan `my_val` di folder repositori. Jika `my_val` sudah ada, skrip memakai kembali split tersebut; pastikan keduanya berasal dari dataset dan seed yang dimaksud. Bobot awal ImageNet diunduh saat pertama kali diperlukan. Checkpoint dengan akurasi validasi terbaik disimpan ke `checkpoints/percobaan1.pth`.

### 4. Siapkan submission dan evaluasi

Cadangkan bobot submission sebelumnya jika ada, kemudian salin checkpoint hasil training:

```powershell
if ((Test-Path ".\submission\model.pth") -and !(Test-Path ".\checkpoints\model_lama.pth")) {
    Copy-Item ".\submission\model.pth" ".\checkpoints\model_lama.pth"
}
Copy-Item ".\checkpoints\percobaan1.pth" ".\submission\model.pth"
.\.venv\Scripts\python.exe evaluate.py --submission ".\submission" --data ".\my_val" --out "checkpoints\evaluasi_percobaan1.json"
```

Evaluator memuat `submission/model.pth`. Salinan checkpoint harus diperbarui sebelum mengevaluasi hasil training baru. Untuk mengevaluasi bobot yang sudah disertakan dalam repo, cukup jalankan evaluator setelah environment dan split validasi tersedia. Training ulang dapat menghasilkan sedikit perbedaan skor antarperangkat atau versi pustaka.

## Struktur Repositori

```
README.md
experiments.md      # catatan eksperimen kelompok
train.py            # pelatihan dan pembagian data lokal
evaluate.py         # skrip evaluasi resmi
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model untuk pengumpulan
checkpoints/        # checkpoint dan hasil evaluasi lokal, diabaikan Git
my_train/           # split training lokal, diabaikan Git
my_val/             # split validasi lokal, diabaikan Git
```
