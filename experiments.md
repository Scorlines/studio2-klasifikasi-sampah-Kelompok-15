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
