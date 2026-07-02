# 📝 Panduan Presentasi - Chatbot Toko Sembako Barokah

## 🎯 Opening (1-2 menit)

### Perkenalan
> "Assalamualaikum, selamat pagi/siang. Saya Muhamad Abdul Aziz, hari ini saya akan mempresentasikan proyek akhir mata kuliah Data Mining, yaitu **Chatbot Toko Sembako Barokah** - sebuah asisten virtual berbasis NLP yang bisa membantu pelanggan toko sembako mendapatkan informasi harga, cara pesan, layanan antar, dan pertanyaan lainnya."

### Latar Belakang
> "Toko sembako biasanya menerima banyak pertanyaan berulang dari pelanggan - harga beras berapa? bisa antar gak? buka jam berapa? Chatbot ini dibuat untuk menangani pertanyaan-pertanyaan tersebut secara otomatis, 24 jam, dengan respons yang natural."

---

## 🏗️ Arsitektur Sistem (3-4 menit)

### Dua Lapis Pemahaman
> "Chatbot ini menggunakan **dua lapis pemahaman** yang saling melengkapi:"

**1. Layer Leksikal (TF-IDF)**
- Preprocessing: case folding, normalisasi slang, stopword removal, stemming Sastrawi
- Ekstraksi fitur: TF-IDF word + character n-gram
- Classifier: Supervised learning (dibandingkan 3 algoritma)

**2. Layer Semantik (Sentence Embedding)** - *opsional*
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Fungsi: Menangkap makna kalimat walau tidak ada kata yang sama
- Digunakan sebagai "gate" untuk deteksi out-of-scope

### Pipeline Prediksi
> "Saat user mengirim pesan, sistem melakukan:"

```
Input → Preprocessing → TF-IDF Classifier → Cascade Rules → Response
                              ↓
                    [Confidence check]
                              ↓
                    [Semantic validation]
                              ↓
                    [Context memory]
```

**Cascade Rules (R1-R6):**
1. **R1**: Classifier yakin (≥55%) → langsung jawab
2. **R2**: Classifier lemah tapi semantik kuat → pakai embedding
3. **R3**: Confidence tipis (selisih <10%) → minta klarifikasi
4. **R4**: Out-of-scope detected → jawab sopan
5. **R5**: Follow-up pendek → warisi konteks sebelumnya
6. **R6**: Fallback → minta user ulang dengan kalimat lain

---

## 📊 Dataset (1-2 menit)

> "Dataset yang digunakan:"
- **24 intent** (kategori pertanyaan)
- **~200 pattern** (contoh kalimat latih)
- Topik: harga sembako, pemesanan, pembayaran, komplain, dll

**Contoh Intent:**
| Intent | Contoh Pattern |
|--------|----------------|
| `harga_beras` | "Beras berapa harganya?", "Harga beras sekarang?" |
| `cara_pesan` | "Gimana cara ordernya?", "Mau beli lewat mana?" |
| `layanan_antar` | "Bisa antar ke rumah gak?", "Ada delivery?" |

---

## 🔬 Perbandingan Model (2-3 menit)

> "Saya membandingkan 3 algoritma classifier menggunakan **5-fold cross-validation**:"

### Algoritma yang Dibandingkan:
1. **Multinomial Naive Bayes** - Probabilistik, cepat
2. **Logistic Regression** - Linear, interpretable
3. **SVM (Linear Kernel)** - Margin maksimal, akurat

### Cara Membaca Chart:
> "Chart ini menunjukkan akurasi rata-rata dari 5 kali validasi:"
- **Bar ungu gelap** = model terpilih (akurasi tertinggi)
- **Error bar** = standar deviasi (konsistensi)
- Model dengan error bar pendek = lebih stabil

### Mengapa Cross-Validation?
> "Dataset kecil rentan bias kalau cuma satu kali train/test split. Cross-validation membagi data jadi 5 bagian, rotasi 5 kali, lalu dirata-rata. Hasilnya lebih reliable."

---

## 📈 Evaluasi Model (3-4 menit)

### 1. Confusion Matrix
> "Heatmap ini menunjukkan prediksi vs aktual:"
- **Diagonal (ungu gelap)** = prediksi benar
- **Sel off-diagonal** = kesalahan prediksi
- Hover untuk lihat detail

**Cara menjelaskan:**
> "Semakin gelap diagonal, semakin bagus. Kalau ada warna di luar diagonal, berarti ada intent yang sering tertukar."

### 2. F1-Score per Intent
> "Bar chart F1-score menunjukkan performa per kategori:"
- **Hijau** = F1 tinggi (≥80%)
- **Kuning** = F1 sedang (60-80%)
- **Merah** = F1 rendah (<60%)

**Jika ada intent dengan F1 rendah:**
> "Intent X agak sulit dibedakan karena pattern-nya mirip dengan intent Y. Tapi secara keseluruhan performa sudah baik."

---

## 🧪 Uji Coba (5-6 menit)

### 1. Uji Paraphrase
**Tujuan:** Chatbot harus mengenali kalimat yang **berbeda kata tapi sama makna**

**Contoh kasus:**
| Input User | Expected | Result |
|------------|----------|--------|
| "Mau order gimana caranya" | cara_pesan | ✅ PASS |
| "Boleh kirim ke rumah gak" | layanan_antar | ✅ PASS |
| "Beras nya berapa duit" | harga_beras | ✅ PASS |

> "Lihat gauge di kanan - menunjukkan persentase kelulusan. Target ≥80%."

### 2. Uji Slang & Typo
**Tujuan:** Chatbot harus paham **bahasa chat sehari-hari** (singkatan, typo, casual)

**Contoh kasus:**
| Input User | Expected | Result |
|------------|----------|--------|
| "beras brp duitnya" | harga_beras | ✅ PASS |
| "cara pesen gmn" | cara_pesan | ✅ PASS |
| "bs anter ga ke rumah" | layanan_antar | ✅ PASS |

> "Sistem normalisasi slang mengubah 'brp' → 'berapa', 'gmn' → 'bagaimana', dll."

### 3. Uji Memori Konteks *(jika embedding aktif)*
**Tujuan:** Chatbot harus ingat **topik sebelumnya** saat user bertanya singkat

**Contoh:**
```
User: "Beras berapa harganya?"
Bot:  "Harga beras kami: ..."
User: "Kalau yang premium?"        ← follow-up singkat
Bot:  "Untuk beras premium..."     ← ingat konteks beras
```

### 4. Uji Out-of-Scope
**Tujuan:** Chatbot harus **menolak dengan sopan** pertanyaan di luar topik

**Contoh:**
| Input User | Expected | Result |
|------------|----------|--------|
| "siapa presiden indonesia" | out_of_scope | ✅ PASS |
| "cuaca hari ini gimana" | out_of_scope | ✅ PASS |
| "berapa 25 kali 4" | out_of_scope | ✅ PASS |

### 5. Uji Basa-basi (Small-talk)
**Tujuan:** Chatbot harus bisa **basa-basi natural**

**Contoh:**
| Input User | Expected | Result |
|------------|----------|--------|
| "kamu siapa" | identitas_bot | ✅ PASS |
| "apa kabar" | apa_kabar | ✅ PASS |
| "bisa bantu apa aja" | bantuan | ✅ PASS |

---

## 💻 Demo Live (3-4 menit)

### Skenario Demo yang Disarankan:

**1. Tanya Harga (basic)**
```
User: "Beras berapa harganya?"
→ Lihat meta badge: intent, confidence, method
```

**2. Slang/Typo (advanced)**
```
User: "brp hrga berasnya min"
→ Sistem tetap paham meski banyak singkatan
```

**3. Follow-up dengan konteks**
```
User: "Harga minyak goreng?"
Bot:  "..."
User: "Kalau yang curah?"
→ Bot ingat masih bahas minyak goreng
```

**4. Out-of-scope**
```
User: "Siapa Jokowi?"
→ Bot menjawab sopan bahwa itu di luar topik
```

**5. Basa-basi**
```
User: "Halo, kamu siapa?"
→ Bot memperkenalkan diri
```

**6. Klarifikasi (jika muncul)**
```
User: "Bisa kirim?"
→ Bot mungkin tanya: "Maksud Anda layanan antar atau ongkos kirim?"
→ Pilih salah satu opsi
```

---

## 🎨 UI/UX Features (1-2 menit)

> "Dashboard ini juga punya beberapa fitur UX:"

1. **Typing Effect** - Jawaban bot muncul bertahap seperti orang mengetik
2. **Auto-focus** - Input field tetap fokus setelah kirim pesan
3. **Glassmorphism Design** - UI modern dengan efek kaca blur
4. **Interactive Charts** - Plotly chart yang bisa di-zoom dan hover
5. **Dark/Light Theme** - Bisa disesuaikan via config

---

## 🔧 Tech Stack (30 detik)

| Komponen | Teknologi |
|----------|-----------|
| Framework | Streamlit |
| NLP | scikit-learn, Sastrawi |
| Embedding | sentence-transformers |
| Visualization | Plotly |
| Language | Python |

---

## ❓ Antisipasi Pertanyaan

### Q: "Kenapa akurasi cuma XX%?"
**A:** "Dataset relatif kecil (~200 pattern, 24 intent). Dengan cross-validation, akurasi XX% ± Y% sudah reasonable. Untuk production, dataset bisa ditambah."

### Q: "Kenapa pakai TF-IDF, bukan deep learning?"
**A:** "TF-IDF + classifier tradisional lebih cepat, ringan, dan interpretable untuk dataset kecil. Deep learning butuh data ribuan contoh."

### Q: "Embedding kenapa opsional?"
**A:** "Model embedding ~120MB, berat untuk environment tanpa GPU. TF-IDF sudah cukup akurat. Embedding dipakai sebagai 'second opinion' dan gate out-of-scope."

### Q: "Bagaimana cara menambah intent baru?"
**A:** "Edit `dataset_sembako.json`, tambah intent dengan pattern-nya, lalu retrain model. Tidak perlu ubah code."

### Q: "Chatbot ini bisa deploy dimana?"
**A:** "Gratis: Streamlit Community Cloud, Hugging Face Spaces. Bisa juga di-render.com atau Railway.app."

---

## 🎬 Closing (30 detik)

> "Demikian presentasi saya tentang Chatbot Toko Sembako Barokah. Chatbot ini bisa mengenali 24 jenis pertanyaan pelanggan, paham bahasa slang dan typo, punya memori konteks, dan menolak pertanyaan di luar topik dengan sopan. Semua evaluasi dan uji coba bisa dilihat langsung di tab Model & Akurasi. Terima kasih, ada pertanyaan?"

---

## 📌 Tips Saat Presentasi

1. **Mulai dari Demo** - Langsung tunjukkan chatbot bekerja, baru jelaskan teknikal
2. **Hover Chart** - Gunakan hover pada Plotly chart untuk menunjukkan detail
3. **Bandingkan** - "Lihat, meski user ketik 'brp' bukan 'berapa', sistem tetap paham"
4. **Jujur soal Limitasi** - "Dataset kecil, tapi dengan cross-validation kita dapat estimasi yang reliable"
5. **Siapkan Backup** - Screenshot hasil uji jika live demo bermasalah

---

*Good luck presentasinya! 🚀*
