# Log Eksperimen — Studio 2 Klasifikasi Sampah (Kelompok 15)

Catatan seluruh eksperimen pelatihan model untuk klasifikasi 5 kelas sampah:
1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

---

## Tabel Ringkasan Eksperimen

| ID | Tanggal | Arsitektur / Backbone | Resolusi | Epoch | Batch Size | LR & Scheduler | Augmentasi | Akurasi | Macro-F1 | F1 Kertas Kardus (Min) | Catatan / Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| **EXP-00** | 12-10-2026 | ResNet-18 (ImageNet Pretrained) | 224x224 | 8 | 32 | AdamW (3e-4), Cosine | RRCrop, HFlip, ColorJitter | 93.17% | **0.9279** | 0.8738 | **Baseline (Preview 1)** ✅ |
| **EXP-01** | 13-10-2026 | ResNet-18 (ImageNet Pretrained) | 224x224 | 8 | 16 | AdamW (3e-4), Cosine | RRCrop, HFlip, ColorJitter | 93.66% | **0.9330** | 0.8952 | percobaan1; submission terbaru untuk Preview 1 |

Semua skor di tabel merupakan hasil validasi lokal. Skor resmi Preview 1 pada validation set tersembunyi belum diumumkan.

---

## Detail Eksperimen

### EXP-00: Baseline ResNet-18 Pretrained (Preview 1)
- **Komit/Branch**: `37af361` (main)
- **Detail Metrik (evaluasi pada `my_val` 410 citra)**:
  - Macro-F1: `0.9279`
  - Akurasi: `93.17%`
  - F1 `botol_plastik`: `0.9536`
  - F1 `gelas_plastik`: `0.9487`
  - F1 `kertas_kardus`: `0.8738` *(paling rawan tertukar dengan saset/styrofoam)*
  - F1 `saset_kemasan`: `0.9261`
  - F1 `styrofoam`: `0.9372`
- **Confusion Matrix**:
  ```text
                  pred: botol  gelas  kertas  saset  styrofoam
  botol_plastik          72      1       0      3          2
  gelas_plastik           1     74       0      1          2
  kertas_kardus           0      2      45      4          3
  saset_kemasan           0      1       2     94          3
  styrofoam               0      0       2      1         97
  ```
- **Kesimpulan**:
  - Baseline berjalan sempurna dan melebihi target acuan awal (~0.92).
  - Titik lemah ada di kelas `kertas_kardus` (data paling sedikit, F1 0.8738).

### EXP-01: ResNet-18, Batch Size 16 (percobaan1)

- **Komit model dan README / branch**: `f44f017` (`main`)
- **Data**: dataset resmi, tanpa data tambahan; 2.343 gambar training dan 410 gambar validasi lokal. Split sekitar 85%/15% per kelas, seed `42`.
- **Konfigurasi**: 8 epoch, batch size `16`, AdamW (`lr=3e-4`, `weight_decay=1e-4`), CosineAnnealingLR (`T_max=8`), CrossEntropyLoss (`label_smoothing=0.1`).
- **Augmentasi training**: RandomResizedCrop(224, scale=(0.6, 1.0)), RandomHorizontalFlip, ColorJitter (brightness/contrast/saturation=0.2), normalisasi ImageNet.
- **Transform validasi**: Resize(256), CenterCrop(224), normalisasi ImageNet.
- **Perangkat**: NVIDIA GeForce RTX 3050 dengan CUDA, VRAM 4 GB, Windows.
- **Checkpoint terpilih**: epoch 8, berdasarkan akurasi validasi tertinggi; disimpan ke `checkpoints/percobaan1.pth` dan disalin ke `submission/model.pth` sebelum evaluasi.
- **Sumber metrik**: hasil evaluator lokal `checkpoints/evaluasi_percobaan1.json` (folder `checkpoints` diabaikan Git).

Perkembangan training:

| Epoch | Train Loss | Akurasi Validasi |
|---|---:|---:|
| 1 | 0.9312 | 0.8610 |
| 2 | 0.7050 | 0.8829 |
| 3 | 0.6301 | 0.9024 |
| 4 | 0.5441 | 0.9195 |
| 5 | 0.4994 | 0.9244 |
| 6 | 0.4692 | 0.9293 |
| 7 | 0.4562 | 0.9268 |
| 8 | 0.4424 | **0.9366** |

Hasil evaluasi checkpoint terpilih:

| Metrik | Nilai |
|---|---:|
| Macro-F1 | **0.9330** |
| Akurasi | **93.66%** |
| F1 botol_plastik | 0.9677 |
| F1 gelas_plastik | 0.9150 |
| F1 kertas_kardus | 0.8952 |
| F1 saset_kemasan | 0.9458 |
| F1 styrofoam | 0.9412 |

Confusion matrix (baris = label asli, kolom = prediksi):

```text
                pred: botol  gelas  kertas  saset  styrofoam
botol_plastik            75      1       0      2         0
gelas_plastik             1     70       2      1         4
kertas_kardus             0      3      47      2         2
saset_kemasan             1      0       1     96         2
styrofoam                0      1       1      2        96
```

Perintah reproduksi dari folder repo (PowerShell; dataset berada di `../train` dan environment sudah siap):

```powershell
.\.venv\Scripts\python.exe -u train.py --data "..\train" --epochs 8 --batch-size 16 --lr 3e-4 --seed 42 --out "checkpoints\percobaan1.pth"
Copy-Item ".\checkpoints\percobaan1.pth" ".\submission\model.pth"
.\.venv\Scripts\python.exe evaluate.py --submission ".\submission" --data ".\my_val" --out "checkpoints\evaluasi_percobaan1.json"
```

Catatan hasil:

- Macro-F1 lokal meningkat dari catatan baseline `0.9279` menjadi `0.9330`. Perbedaan satu percobaan belum cukup untuk menyimpulkan bahwa batch size 16 selalu lebih baik.
- `kertas_kardus` tetap memiliki F1 terendah, meskipun meningkat dari `0.8738` menjadi `0.8952`.
- F1 `gelas_plastik` turun dari `0.9487` menjadi `0.9150`; kesalahan terbanyak pada kelas ini adalah 4 gambar yang diprediksi sebagai `styrofoam`.
- Model berhasil dijalankan oleh evaluator lokal. Skor resmi Preview 1 menunggu evaluasi asisten.
- Rencana berikutnya (belum dilakukan): periksa contoh kesalahan `kertas_kardus` dan `gelas_plastik`, lalu uji satu perubahan konfigurasi pada split validasi yang sama.
