"""
Melatih & membandingkan beberapa classifier intent (TF-IDF word+char n-gram) dari
dataset_sembako.json, mengevaluasi akurasinya (train/test split + cross-validation),
dan menyediakan semua artefak yang dibutuhkan dashboard Streamlit (app.py).
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

import chatbot_engine as engine

# SVC(probability=True) memakai Platt scaling untuk predict_proba -- scikit-learn >= 1.9
# menandainya deprecated (disarankan CalibratedClassifierCV) tapi masih berfungsi penuh.
# Diredam supaya output self-test tetap bersih; tidak memengaruhi hasil apa pun.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*parameter was deprecated.*")

DATASET_PATH = Path(__file__).parent / "dataset_sembako.json"


def compare_classifiers(X, y, cv):
    """Latih & evaluasi beberapa classifier supervised via cross-validation yang sama,
    kembalikan tabel perbandingan + model yang sudah di-fit ulang di seluruh data."""
    candidates = {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (Linear Kernel)": SVC(kernel="linear", probability=True,
                                    class_weight="balanced", random_state=42),
    }
    rows = []
    fitted = {}
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y, cv=cv)
        clf_final = clone(clf)
        clf_final.fit(X, y)
        fitted[name] = clf_final
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})

    best_name = max(rows, key=lambda r: r["cv_mean"])["model"]
    return {"comparison_table": rows, "fitted_models": fitted, "best_model_name": best_name}


def compare_embedding_classifiers(pattern_embeddings, y, cv):
    """Bandingkan classifier berbasis EMBEDDING (NearestCentroid & kNN cosine) via CV
    yang sama dengan jalur TF-IDF -- untuk menunjukkan nilai tambah layer semantik.
    Tidak memengaruhi classifier produksi (tetap yang terbaik dari compare_classifiers)."""
    # Embedding sudah L2-normalized, jadi jarak euclidean di NearestCentroid setara monoton
    # dengan cosine (euclidean^2 = 2 - 2*cosine). kNN memakai metric cosine langsung.
    candidates = {
        "Embedding + NearestCentroid": NearestCentroid(),
        "Embedding + kNN (k=3, cosine)": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
    }
    rows = []
    for name, clf in candidates.items():
        scores = cross_val_score(clf, pattern_embeddings, y, cv=cv)
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
    return rows


def train_and_evaluate(dataset_path=DATASET_PATH):
    dataset = engine.load_dataset(dataset_path)
    texts, tags = engine.build_training_rows(dataset)

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)

    X_word = word_vectorizer.fit_transform(texts)
    X_char = char_vectorizer.fit_transform(texts)
    X = hstack([X_word, X_char]).tocsr()
    y = np.array(tags)
    labels = sorted(set(tags))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perbandingan classifier (dipakai baik untuk memilih model final maupun ditampilkan
    # di dashboard sebagai bukti analisis akurasi antar-algoritma).
    comparison = compare_classifiers(X, y, cv)
    best_model_name = comparison["best_model_name"]

    # Evaluasi test-split & confusion matrix/classification report memakai model TERBAIK,
    # supaya semua metrik yang tampil di dashboard konsisten dengan model yang benar-benar dipakai chat.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    split_classifier = clone(comparison["fitted_models"][best_model_name])
    split_classifier.fit(X_train, y_train)
    y_pred = split_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    best_row = next(r for r in comparison["comparison_table"] if r["model"] == best_model_name)

    # ---- Layer semantik (opsional & aman kalau gagal) ----
    embedder = engine.load_embedder()
    embedding_available = embedder is not None
    pattern_embeddings = centroid_matrix = centroid_tags = None
    embedding_comparison = []
    if embedding_available:
        pattern_embeddings, emb_tags = engine.build_pattern_embeddings(dataset, embedder)
        centroid_matrix, centroid_tags = engine.build_centroids(pattern_embeddings, emb_tags)
        embedding_comparison = compare_embedding_classifiers(pattern_embeddings, np.array(emb_tags), cv)

    return {
        "dataset": dataset,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "classifier": comparison["fitted_models"][best_model_name],
        "pattern_vectors": X,
        "pattern_tags": tags,
        "labels": labels,
        "accuracy": accuracy,
        "cv_mean": best_row["cv_mean"],
        "cv_std": best_row["cv_std"],
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classifier_comparison": comparison["comparison_table"],
        "best_model_name": best_model_name,
        "n_intents": len(dataset),
        "n_samples": len(texts),
        # Artefak semantik (None/[] kalau embedding tidak tersedia -> jalur TF-IDF lama).
        "embedder": embedder,
        "embedding_available": embedding_available,
        "pattern_embeddings": pattern_embeddings,
        "centroid_matrix": centroid_matrix,
        "centroid_tags": centroid_tags,
        "embedding_comparison": embedding_comparison,
    }


def _predict(artifacts, text, prev_context=None):
    """Pembungkus predict_intent yang otomatis mengoper artefak embedding (kalau ada)."""
    return engine.predict_intent(
        text,
        artifacts["word_vectorizer"],
        artifacts["char_vectorizer"],
        artifacts["classifier"],
        artifacts["pattern_vectors"],
        artifacts["pattern_tags"],
        embedder=artifacts.get("embedder"),
        centroid_matrix=artifacts.get("centroid_matrix"),
        centroid_tags=artifacts.get("centroid_tags"),
        prev_context=prev_context,
    )


def _self_test_exact_match(artifacts):
    """Uji tiap intent dengan satu pattern verbatim dari dataset -- harus selalu match."""
    print("\n=== Uji Exact-Match (tiap intent, 1 pattern verbatim) ===")
    passed = 0
    for intent in artifacts["dataset"]:
        sample = intent["patterns"][0]
        result = _predict(artifacts, sample)
        ok = result["tag"] == intent["tag"]
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] '{sample}' -> expected={intent['tag']} got={result['tag']} "
              f"(conf={result['confidence']:.2f}, {result['method']})")
    total = len(artifacts["dataset"])
    print(f"\nExact-match: {passed}/{total} lolos.")


# Sebagian kasus punya lebih dari satu tag yang "benar" karena makna intent-nya
# memang berdekatan (ambiguitas wajar, bukan kesalahan model) -- dicatat sebagai
# tuple tag yang diterima, bukan cuma satu tag.
PARAPHRASE_CASES = [
    ("Mau order gimana caranya", ("cara_pesan",)),
    ("Tokonya buka jam segini gak", ("jam_operasional",)),
    ("Boleh kirim ke rumah gak", ("layanan_antar", "ongkos_kirim")),
    ("Beras nya berapa duit", ("harga_beras",)),
    ("Bisa transfer gak bayarnya", ("metode_bayar",)),
    ("Ada potongan kalau beli banyak", ("promo_diskon", "harga_grosir")),
    ("Barang saya kok gak sesuai ya", ("komplain_retur",)),
    ("Ini toko dimana ya lokasinya", ("lokasi_toko",)),
    ("Minta nomor kontaknya dong", ("kontak_toko",)),
    ("Boleh nyicil bayarnya gak", ("jenis_pembayaran_hutang",)),
]

# Kasus yang secara eksplisit menguji slang, typo, dan kalimat casual/tidak lengkap --
# target pass-rate realistis >= 80% (bukan 100%, kombinasi slang+typo ekstrem selalu
# punya kasus tepi).
SLANG_TYPO_CASES = [
    ("beras brp duitnya", ("harga_beras",)),
    ("harga bearas berapa", ("harga_beras",)),
    ("ada beras ga", ("harga_beras", "cek_stok")),
    ("minyk goreng ad ga", ("harga_minyak", "cek_stok")),
    ("cara pesen gmn", ("cara_pesan",)),
    ("pesenin dong barangnya", ("cara_pesan",)),
    ("bs anter ga ke rumah", ("layanan_antar",)),
    ("ongkirnya brp", ("ongkos_kirim",)),
    ("bayar pake qris bs ga", ("metode_bayar",)),
    ("ad diskon ga klo beli byk", ("promo_diskon", "harga_grosir")),
    ("barang gw rusak nih gmn", ("komplain_retur",)),
    ("bs ngutang dlu ga", ("jenis_pembayaran_hutang",)),
    ("toko buka jam brp", ("jam_operasional",)),
    ("lokasinya dmn ya", ("lokasi_toko",)),
    ("nomor wa toko brp", ("kontak_toko",)),
    ("min ada gula ga", ("harga_gula", "cek_stok")),
    ("mau jd reseller gmn caranya", ("jadi_reseller",)),
    ("kualitas berasnya bgs ga", ("kualitas_produk",)),
]


def _evaluate_cases(artifacts, cases):
    rows = []
    for text, expected_tags in cases:
        result = _predict(artifacts, text)
        rows.append({
            "input": text,
            "expected_tag": "/".join(expected_tags),
            "predicted_tag": result["tag"],
            "confidence": result["confidence"],
            "method": result["method"],
            "status": "PASS" if result["tag"] in expected_tags else "FAIL",
        })
    return rows


def evaluate_paraphrase_cases(artifacts, cases=PARAPHRASE_CASES):
    """Jalankan predict_intent untuk tiap kasus paraphrase, kembalikan list of dict
    (dipakai baik oleh self-test terminal maupun tabel 'uji paraphrase' di dashboard)."""
    return _evaluate_cases(artifacts, cases)


def evaluate_slang_typo_cases(artifacts, cases=SLANG_TYPO_CASES):
    """Sama seperti evaluate_paraphrase_cases, tapi untuk kasus slang/typo/casual."""
    return _evaluate_cases(artifacts, cases)


# Uji MEMORI KONTEKS: (setup, follow-up, tag_yang_diterima_utk_followup).
# Kasus terakhir adalah NEGATIVE GUARD: follow-up menyebut topik baru -> harus PINDAH,
# bukan mewarisi topik lama.
CONTEXT_FOLLOWUP_CASES = [
    ("beras berapa harganya", "kalau yang premium?", ("harga_beras",)),
    ("harga minyak goreng", "kalo yang curah gimana", ("harga_minyak",)),
    ("mau pesan gimana", "kalau lewat wa?", ("cara_pesan",)),
    ("ongkir ke rumah berapa", "yang gratis gimana", ("ongkos_kirim",)),
    ("bisa jadi reseller?", "terus caranya?", ("jadi_reseller",)),
    ("harga gula berapa", "yang merah?", ("harga_gula",)),
    ("ada layanan antar?", "kalau ke luar kota?", ("layanan_antar", "ongkos_kirim")),
    ("harga telur", "yang kampung brp", ("harga_telur",)),
    ("harga beras berapa", "kalau minyak goreng gimana?", ("harga_minyak",)),  # negative guard
]

# Uji OUT-OF-SCOPE: harus menghasilkan type == "out_of_scope".
OOS_CASES = [
    "cuaca hari ini gimana",
    "siapa presiden indonesia",
    "ceritain film bagus dong",
    "berapa 25 kali 4",
    "kamu suka main game apa",
    "rekomendasi tempat wisata dong",
    "gimana cara bikin website",
]

# Uji BASA-BASI (small-talk): harus dijawab dengan intent yang benar.
SMALLTALK_CASES = [
    ("kamu siapa", ("identitas_bot",)),
    ("ini bot ya", ("identitas_bot",)),
    ("km robot bukan sih", ("identitas_bot",)),
    ("bisa bantu apa aja", ("bantuan",)),
    ("kamu bisa apa", ("bantuan",)),
    ("apa kabar", ("apa_kabar",)),
    ("pa kabar min", ("apa_kabar",)),
]


def evaluate_context_cases(artifacts, cases=CONTEXT_FOLLOWUP_CASES):
    """Uji 2-turn: proses setup, update konteks, lalu proses follow-up dengan konteks itu."""
    rows = []
    for setup, followup, expected_tags in cases:
        setup_res = _predict(artifacts, setup)
        ctx = engine.update_context(None, setup_res, setup)
        follow_res = _predict(artifacts, followup, prev_context=ctx)
        rows.append({
            "input": f"[{setup}] -> {followup}",
            "expected_tag": "/".join(expected_tags),
            "predicted_tag": follow_res["tag"],
            "confidence": follow_res["confidence"],
            "method": follow_res["method"],
            "status": "PASS" if follow_res["tag"] in expected_tags else "FAIL",
        })
    return rows


def evaluate_oos_cases(artifacts, cases=OOS_CASES):
    rows = []
    for text in cases:
        result = _predict(artifacts, text)
        ok = result["type"] == "out_of_scope"
        rows.append({
            "input": text,
            "expected_tag": "out_of_scope",
            "predicted_tag": result["type"],
            "confidence": result["confidence"],
            "method": result["method"],
            "status": "PASS" if ok else "FAIL",
        })
    return rows


def evaluate_smalltalk_cases(artifacts, cases=SMALLTALK_CASES):
    return _evaluate_cases(artifacts, cases)


def _print_case_results(title, rows):
    print(f"\n=== {title} ===")
    for row in rows:
        print(f"[{row['status']}] '{row['input']}' -> expected={row['expected_tag']} "
              f"got={row['predicted_tag']} (conf={row['confidence']:.2f}, {row['method']})")
    passed = sum(1 for row in rows if row["status"] == "PASS")
    print(f"\n{title}: {passed}/{len(rows)} lolos.")


def _self_test_paraphrase(artifacts):
    _print_case_results("Uji Paraphrase (kalimat TIDAK identik dengan data latih)",
                         evaluate_paraphrase_cases(artifacts))


def _self_test_slang_typo(artifacts):
    _print_case_results("Uji Slang & Typo (chat alami manusia)",
                         evaluate_slang_typo_cases(artifacts))


def _self_test_context(artifacts):
    _print_case_results("Uji Memori Konteks (follow-up 2-turn)",
                        evaluate_context_cases(artifacts))


def _self_test_oos(artifacts):
    _print_case_results("Uji Out-of-Scope (di luar topik)",
                        evaluate_oos_cases(artifacts))


def _self_test_smalltalk(artifacts):
    _print_case_results("Uji Basa-basi (small-talk)",
                        evaluate_smalltalk_cases(artifacts))


if __name__ == "__main__":
    artifacts = train_and_evaluate()
    print(f"Dataset: {artifacts['n_intents']} intent, {artifacts['n_samples']} sample pattern.")
    print(f"Embedding aktif: {artifacts['embedding_available']}")

    print("\n=== Perbandingan Algoritma Classifier TF-IDF (5-fold CV) ===")
    for row in artifacts["classifier_comparison"]:
        marker = " <- terpilih (produksi)" if row["model"] == artifacts["best_model_name"] else ""
        print(f"{row['model']:24s} cv_mean={row['cv_mean']:.2%}  cv_std={row['cv_std']:.2%}{marker}")

    if artifacts["embedding_comparison"]:
        print("\n=== Perbandingan Classifier berbasis Embedding (5-fold CV yang sama) ===")
        for row in artifacts["embedding_comparison"]:
            print(f"{row['model']:32s} cv_mean={row['cv_mean']:.2%}  cv_std={row['cv_std']:.2%}")

    print(f"\nModel terpilih: {artifacts['best_model_name']}")
    print(f"Akurasi (test split 25%, model terpilih): {artifacts['accuracy']:.2%}")
    print(f"Akurasi (5-fold CV, model terpilih): {artifacts['cv_mean']:.2%} (+/- {artifacts['cv_std']:.2%})")

    _self_test_exact_match(artifacts)
    _self_test_paraphrase(artifacts)
    _self_test_slang_typo(artifacts)
    _self_test_context(artifacts)
    _self_test_oos(artifacts)
    _self_test_smalltalk(artifacts)
