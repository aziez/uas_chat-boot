# ChatBot "Toko Sembako Barokah"

UAS Data Mining / Pembuatan ChatBot / NLP -- Muhamad Abdul Aziz
Tema: **Penjualan Sembako**

## Pendekatan

Chatbot dua lapis: **leksikal** (TF-IDF + classifier supervised, pemutus utama & bagian yang
dinilai) dan **semantik** (sentence-embeddings, untuk paham makna + deteksi di-luar-topik).
Ditambah **memori konteks** dan **klarifikasi** supaya tidak kaku dan lebih paham percakapan.

1. **Preprocessing**: case folding -> hapus tanda baca -> **normalisasi slang/singkatan chat**
   (gak/gmn/brp/sy/yg/dgn/dll -> bentuk baku, via kamus `SLANG_DICT`) -> stopword removal ->
   **stemming Bahasa Indonesia** (PySastrawi, menyamakan "membeli"/"dibeli"/"pembelian" -> "beli").
   Untuk jalur embedding dipakai `clean_text_semantic()` (normalisasi slang saja, TANPA
   stopword/stemming) supaya kalimat tetap utuh & natural buat transformer.
2. **Fitur leksikal**: TF-IDF kata (unigram+bigram) **digabung** dengan TF-IDF karakter
   (`char_wb`, n-gram 3-5) supaya tahan typo.
3. **Classifier**: 3 algoritma supervised (Multinomial Naive Bayes, Logistic Regression, SVM
   kernel linear) dibandingkan via 5-fold cross-validation, terbaik dipilih otomatis sebagai
   model produksi. Sebagai pembanding, classifier berbasis embedding (NearestCentroid & kNN
   cosine) juga dievaluasi di fold yang sama -- semua tampil di dashboard.
4. **Layer semantik (offline)**: `sentence-transformers` model
   `paraphrase-multilingual-MiniLM-L12-v2` meng-encode tiap intent jadi centroid. Saat inference,
   cascade R1-R6 memadukan confidence classifier (p1) dengan similarity semantik (s1):
   - classifier yakin -> dipercaya; classifier lemah tapi semantik kuat -> **semantic fallback**
     (paham walau tak ada kata yang sama); dua-duanya lemah -> **di luar topik**; ragu -> **klarifikasi**.
5. **Memori konteks**: follow-up pendek tanpa subjek (mis. "kalau yang premium?") mewarisi topik
   giliran sebelumnya (`context_carry`), dengan penjaga ketat supaya pertanyaan baru tidak salah warisi.
6. **Basa-basi & out-of-scope**: intent `identitas_bot`/`bantuan`/`apa_kabar` untuk sapaan;
   pertanyaan di luar topik dijawab dengan pengarahan sopan, bukan dipaksa ke intent terdekat.

Hasil self-test (dengan embedding aktif): exact-match **27/27**, paraphrase **10/10**,
slang & typo **18/18**, memori konteks **9/9**, out-of-scope **7/7**, basa-basi **7/7**.

## Cara Menjalankan

```bash\
pip install -r requirements.txt
```

> **Catatan install (penting):** `sentence-transformers` menarik `torch` yang besar. Kalau
> install `torch` terlalu berat/lama di Windows, pasang versi CPU-nya:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

```bash
# (sekali, saat ADA internet) unduh model embedding ke cache lokal:
python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"

# Latih model + jalankan seluruh self-test di terminal
python train_model.py

# Jalankan dashboard (dari dalam folder "2. Coding")
python -m streamlit run app.py
```

> Model embedding di-cache di `C:\Users\<user>\.cache\huggingface\hub` setelah unduhan pertama,
> jadi demo berikutnya jalan **offline penuh** (opsional set `HF_HUB_OFFLINE=1`). Kalau model
> gagal dimuat, chatbot otomatis **turun mulus ke mode TF-IDF** (tidak crash) -- statusnya
> terlihat di sidebar ("Pemahaman semantik: aktif/nonaktif").

> Catatan Windows: kalau `streamlit run app.py` error `'streamlit' is not recognized`, pakai
> `python -m streamlit run app.py` (selalu jalan tanpa utak-atik PATH).

Dashboard punya 2 tab:
- **Chat** -- ngobrol dengan chatbot (loop terus; ketik `keluar`/`selesai`/`bye`/`exit` atau klik
  "Reset percakapan"). Bot ingat konteks, bisa klarifikasi, dan menolak pertanyaan di luar topik dengan sopan.
- **Model & Akurasi** -- perbandingan algoritma (TF-IDF & embedding), akurasi (train/test split +
  5-fold CV), confusion matrix, classification report, dan tabel uji paraphrase / slang-typo /
  memori konteks / out-of-scope / basa-basi.

## File

| File | Fungsi |
|---|---|
| `dataset_sembako.json` | 27 intent, ~295 pattern (formal + slang + typo + casual + basa-basi) |
| `chatbot_engine.py` | Preprocessing (slang + Sastrawi), fitur TF-IDF word+char, layer embedding, cascade prediksi (classifier + semantik + konteks + OOS + klarifikasi) |
| `train_model.py` | Training & perbandingan classifier (TF-IDF: NB/LogReg/SVM; embedding: NearestCentroid/kNN), evaluasi akurasi, seluruh self-test |
| `app.py` | Dashboard Streamlit (Chat + Model & Akurasi) |
