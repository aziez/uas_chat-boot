"""
Mesin chatbot Toko Sembako Barokah.

Dua lapis pemahaman yang saling melengkapi:
1. Leksikal  -> preprocessing (case fold, normalisasi slang, stopword, stemming Sastrawi)
   + TF-IDF kata/karakter + classifier supervised (dilatih di train_model.py).
2. Semantik  -> sentence-embeddings (opsional, offline) untuk menangkap makna kalimat
   walau tidak ada kata yang sama dengan data latih.

predict_intent() memadukan keduanya lewat cascade aturan (R1-R6), menambahkan memori
konteks percakapan, klarifikasi saat ragu, dan deteksi pertanyaan di luar topik.
Semua fungsi murni (tanpa Streamlit) supaya bisa diuji langsung dari train_model.py.
"""

import json
import random
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------------------
# Konstanta preprocessing
# --------------------------------------------------------------------------------------

# Sapaan/panggilan tidak membawa makna intent apa pun -> diperlakukan sebagai stopword.
STOPWORDS = {
    "apakah", "apa", "ada", "bagaimana", "berapa", "dong", "ya",
    "sih", "kah", "yang", "untuk", "di", "ke", "dari", "dan", "atau",
    "ini", "itu", "nya", "saya", "kami", "toko", "anda", "kita",
    "min", "kak", "gan", "sis", "bro", "ka", "kk",
}

# Kamus normalisasi slang/singkatan chat -> bentuk baku, diterapkan per-token.
SLANG_DICT = {
    # negasi
    "gak": "tidak", "ga": "tidak", "g": "tidak", "kaga": "tidak", "kagak": "tidak",
    "nggak": "tidak", "enggak": "tidak", "tdk": "tidak", "gaa": "tidak",
    # kata tanya / gaul
    "gimana": "bagaimana", "gmn": "bagaimana", "gmna": "bagaimana", "gimna": "bagaimana",
    "brp": "berapa", "brapa": "berapa", "berapaan": "berapa", "brepe": "berapa",
    "knp": "kenapa", "kenapa": "kenapa",
    "dmn": "dimana", "dimn": "dimana", "dmana": "dimana",
    "kpn": "kapan", "sapa": "siapa",
    # intensifier
    "bgt": "banget", "bngt": "banget", "gtu": "begitu", "bgtu": "begitu",
    # kata ganti
    "sy": "saya", "gw": "saya", "gue": "saya", "w": "saya", "ane": "saya",
    "lo": "anda", "lu": "anda", "ente": "anda", "qt": "kita",
    # singkatan umum chat
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk",
    "krn": "karena", "karna": "karena", "jd": "jadi", "jg": "juga", "jga": "juga",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "belom": "belum",
    "trs": "terus", "tp": "tapi", "klo": "kalau", "kalo": "kalau",
    "bs": "bisa", "bsa": "bisa", "gpp": "tidak apa apa", "gapapa": "tidak apa apa",
    "sm": "sama", "pd": "pada", "dr": "dari",
    # domain retail/cs -- kata kerja transaksi
    "pesenin": "pesan", "psn": "pesan", "pesen": "pesan", "beliin": "beli",
    "kirimin": "kirim", "kirimn": "kirim", "anterin": "antar", "anter": "antar",
    "bayarin": "bayar", "nawar": "tawar", "nawarin": "tawar",
    # domain retail/cs -- channel/istilah
    "wa": "whatsapp", "tlp": "telepon", "telp": "telepon", "telpon": "telepon",
    "cod": "bayar ditempat", "ongkir": "ongkos kirim",
    # umum
    "mnt": "minta", "tlong": "tolong", "donk": "dong", "nih": "ini", "koq": "kok",
    "minyk": "minyak", "masi": "masih",
}

# --------------------------------------------------------------------------------------
# Konstanta cascade keputusan (di-tuning lewat self-test di train_model.py)
# --------------------------------------------------------------------------------------

# Ambang confidence classifier (proba) & similarity semantik (cosine centroid).
# Nilai dikalibrasi dari distribusi skor nyata (lihat protokol di train_model.py).
# Prinsip: classifier TF-IDF adalah pemutus UTAMA (andal untuk query retail pendek);
# embedding dipakai untuk gate out-of-scope + penyelamat saat classifier lemah, BUKAN
# reranker agresif (embedding cukup noisy pada kalimat pendek/typo).
T_HIGH = 0.55      # classifier dianggap "yakin" di atas ini -> langsung dipercaya
T_LOW = 0.40       # di bawah ini classifier dianggap "lemah"
S_HIGH = 0.62      # similarity semantik dianggap "kuat" di atas ini
MARGIN_CLF = 0.10  # selisih top-2 classifier terlalu tipis -> klarifikasi

# Gate out-of-scope: dua klausa (p1=proba classifier teratas, s1=similarity centroid teratas).
# Sebuah input dianggap DI LUAR TOPIK jika salah satu klausa terpenuhi:
#   (a) classifier lemah DAN semantik tidak kuat, atau
#   (b) classifier sedang tapi semantik sangat rendah (cocok leksikal yang semu, mis. "siapa presiden").
OOS_P_STRICT, OOS_S_STRICT = 0.40, 0.62
OOS_P_LOOSE, OOS_S_LOOSE = 0.60, 0.45

# Memori konteks: follow-up pendek tanpa subjek boleh mewarisi topik giliran sebelumnya.
MAX_TOKENS_CARRY = 4
FOLLOWUP_CUES = {"kalau", "kalo", "klo", "yang", "yg", "terus", "trs", "lalu", "itu", "gimana"}

# Kata kunci topik: kalau input sudah menyebut salah satu, dia pertanyaan mandiri -> JANGAN warisi konteks.
TOPIC_KEYWORDS = {
    "beras", "minyak", "minyakita", "gula", "telur", "telor", "tepung", "terigu",
    "bawang", "cabai", "cabe", "bumbu", "gas", "lpg", "elpiji", "stok",
    "ongkir", "ongkos", "antar", "delivery", "kirim", "grosir", "grosiran", "reseller",
    "promo", "diskon", "bon", "hutang", "utang", "cicil", "bayar", "qris", "transfer",
    "lokasi", "alamat", "jam", "buka", "tutup", "kontak", "komplain", "retur", "kualitas",
}

# Label manusiawi tiap intent -- dipakai untuk teks klarifikasi & pesan out-of-scope.
INTENT_LABELS = {
    "salam": "sapaan", "penutup": "penutup percakapan",
    "jam_operasional": "jam operasional", "lokasi_toko": "lokasi toko",
    "harga_beras": "harga beras", "harga_minyak": "harga minyak goreng",
    "harga_gula": "harga gula", "harga_telur": "harga telur", "harga_tepung": "harga tepung",
    "produk_bumbu": "bumbu dapur", "produk_gas_lpg": "gas LPG", "cek_stok": "ketersediaan stok",
    "cara_pesan": "cara pemesanan", "layanan_antar": "layanan antar",
    "ongkos_kirim": "ongkos kirim", "minimum_order": "minimum order",
    "metode_bayar": "metode pembayaran", "promo_diskon": "promo & diskon",
    "harga_grosir": "harga grosir", "jadi_reseller": "jadi reseller",
    "komplain_retur": "komplain / retur", "kualitas_produk": "kualitas produk",
    "kontak_toko": "kontak toko", "jenis_pembayaran_hutang": "sistem bon / bayar tempo",
    "identitas_bot": "identitas asisten", "bantuan": "bantuan", "apa_kabar": "sapaan",
}

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

UNKNOWN_TAG = "fallback_unknown"
UNKNOWN_RESPONSE = "Maaf, saya belum paham maksud Anda. Bisa diulang dengan kalimat lain?"
OOS_TAG = "out_of_scope"
OOS_RESPONSE = (
    "Maaf, sepertinya itu di luar topik toko. Saya bisa bantu soal harga sembako "
    "(beras, minyak, gula, telur, tepung), bumbu & gas LPG, cek stok, cara pesan, "
    "layanan antar & ongkos kirim, pembayaran, promo/grosir, reseller, dan komplain. "
    "Mau tanya yang mana?"
)

# Singleton -- StemmerFactory().create_stemmer() memuat kamus kata dasar Sastrawi ke
# memori, cukup mahal kalau dipanggil ulang tiap request. Dibuat sekali saat modul di-import.
_STEMMER = StemmerFactory().create_stemmer()


# --------------------------------------------------------------------------------------
# I/O & preprocessing
# --------------------------------------------------------------------------------------

def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_slang(tokens):
    """Petakan tiap token slang/singkatan ke bentuk baku via SLANG_DICT."""
    return [SLANG_DICT.get(tok, tok) for tok in tokens]


def _basic_clean(text):
    """Lowercase + hapus tanda baca + normalisasi spasi. Kembalikan list token ter-normalisasi slang."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize_slang(text.split())


def clean_text(text):
    """Jalur LEKSIKAL (untuk TF-IDF): + hapus stopword + stemming Sastrawi."""
    tokens = [tok for tok in _basic_clean(text) if tok not in STOPWORDS]
    return _STEMMER.stem(" ".join(tokens))


def clean_text_semantic(text):
    """Jalur SEMANTIK (untuk embeddings): normalisasi slang saja, TANPA stopword/stemming --
    transformer butuh kalimat utuh yang natural, bukan hasil stem."""
    return " ".join(_basic_clean(text))


def tokenize(text):
    """Tokenisasi + normalisasi slang, dipakai untuk panel debug di dashboard."""
    return _basic_clean(text)


def build_training_rows(dataset):
    """Ubah setiap pattern di dataset menjadi baris (teks_bersih, tag) untuk jalur TF-IDF."""
    texts, tags = [], []
    for intent in dataset:
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            tags.append(intent["tag"])
    return texts, tags


def get_response(tag, dataset):
    for intent in dataset:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return UNKNOWN_RESPONSE


# --------------------------------------------------------------------------------------
# Layer semantik (embeddings) -- opsional & aman kalau gagal
# --------------------------------------------------------------------------------------

def load_embedder(model_name=EMBED_MODEL_NAME):
    """Muat SentenceTransformer. Kembalikan None kalau gagal (mis. paket/model tidak ada)
    supaya chatbot tetap jalan via jalur TF-IDF (graceful degradation)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


def build_pattern_embeddings(dataset, embedder):
    """Encode semua pattern (jalur semantik), L2-normalized, index-aligned ke tag-nya."""
    texts, tags = [], []
    for intent in dataset:
        for pattern in intent["patterns"]:
            texts.append(clean_text_semantic(pattern))
            tags.append(intent["tag"])
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings, tags


def build_centroids(pattern_embeddings, pattern_tags):
    """Rata-rata embedding per-intent (centroid), dinormalisasi ulang. Prototipe tiap intent."""
    import numpy as np

    by_tag = {}
    for emb, tag in zip(pattern_embeddings, pattern_tags):
        by_tag.setdefault(tag, []).append(emb)

    tags = sorted(by_tag)
    centroids = []
    for tag in tags:
        mean = np.mean(by_tag[tag], axis=0)
        norm = np.linalg.norm(mean)
        centroids.append(mean / norm if norm > 0 else mean)
    return np.vstack(centroids), tags


def _semantic_scores(user_text, embedder, centroid_matrix, centroid_tags):
    """Kembalikan top-2 (tag, similarity) dari cosine query vs centroid tiap intent."""
    import numpy as np

    q = embedder.encode([clean_text_semantic(user_text)], normalize_embeddings=True)[0]
    sims = centroid_matrix @ q  # cosine, karena semua vektor sudah dinormalisasi
    order = np.argsort(sims)[::-1]
    top = [(centroid_tags[i], float(sims[i])) for i in order[:2]]
    while len(top) < 2:
        top.append((None, 0.0))
    return top


# --------------------------------------------------------------------------------------
# Fitur TF-IDF gabungan
# --------------------------------------------------------------------------------------

def build_feature_vector(cleaned_text, word_vectorizer, char_vectorizer):
    """Bangun satu vektor fitur gabungan (word TF-IDF + char n-gram TF-IDF).
    SATU-SATUNYA tempat yang melakukan hstack ini -- dipakai training & inference,
    supaya urutan konkatenasi [word, char] selalu identik di kedua sisi."""
    word_vec = word_vectorizer.transform([cleaned_text])
    char_vec = char_vectorizer.transform([cleaned_text])
    return hstack([word_vec, char_vec]).tocsr()


# --------------------------------------------------------------------------------------
# Konteks percakapan
# --------------------------------------------------------------------------------------

def _has_topic_keyword(cleaned_lexical):
    return any(tok in TOPIC_KEYWORDS for tok in cleaned_lexical.split())


def _should_carry_context(user_text, prev_context, best_proba, decision_type):
    """True hanya kalau SEMUA syarat ketat terpenuhi (anti salah-tebak konteks)."""
    if not prev_context or not prev_context.get("last_intent"):
        return False
    tokens = _basic_clean(user_text)
    if len(tokens) > MAX_TOKENS_CARRY:               # kalimat panjang -> mandiri
        return False
    cleaned_lexical = clean_text(user_text)
    if _has_topic_keyword(cleaned_lexical):          # sudah sebut topik sendiri -> mandiri
        return False
    if not (set(tokens) & FOLLOWUP_CUES):            # tak ada cue follow-up
        return False
    # hanya warisi kalau resolusi giliran ini memang lemah/ragu
    return decision_type in ("clarify", "out_of_scope") or best_proba < T_HIGH


def update_context(prev_context, result, user_text):
    """Kembalikan objek konteks baru. Hanya giliran ber-type 'answer' yang meng-update."""
    if result.get("type") != "answer":
        return prev_context
    tag = result.get("tag")
    if not tag or tag in (OOS_TAG, UNKNOWN_TAG):
        return prev_context
    topic_kw = [tok for tok in clean_text(user_text).split() if tok in TOPIC_KEYWORDS]
    return {
        "last_intent": tag,
        "last_topic_keywords": topic_kw or (prev_context or {}).get("last_topic_keywords", []),
        "last_answer_type": "answer",
    }


# --------------------------------------------------------------------------------------
# Klarifikasi
# --------------------------------------------------------------------------------------

def label_for(tag):
    return INTENT_LABELS.get(tag, tag)


def build_clarify_text(candidates):
    a = label_for(candidates[0]["tag"])
    b = label_for(candidates[1]["tag"])
    return f"Maksud Anda soal **{a}** atau **{b}**? Silakan pilih supaya jawaban saya pas."


def _result(type_, tag, confidence, method, candidates=None, scores=None, context_note=None):
    return {
        "type": type_,
        "tag": tag,
        "confidence": float(confidence),
        "method": method,
        "candidates": candidates or [],
        "scores": scores or {},
        "context_note": context_note,
    }


# --------------------------------------------------------------------------------------
# Prediksi intent (cascade R1-R6 + konteks)
# --------------------------------------------------------------------------------------

def _is_out_of_scope(p1, s1):
    """Gate out-of-scope dua-klausa (lihat konstanta OOS_* & kalibrasi di train_model.py)."""
    return (p1 < OOS_P_STRICT and s1 < OOS_S_STRICT) or (p1 < OOS_P_LOOSE and s1 < OOS_S_LOOSE)


def predict_intent(user_text, word_vectorizer, char_vectorizer, classifier,
                   pattern_vectors, pattern_tags, *,
                   embedder=None, centroid_matrix=None, centroid_tags=None,
                   prev_context=None,
                   t_high=T_HIGH, t_low=T_LOW, s_high=S_HIGH, margin_clf=MARGIN_CLF,
                   threshold=0.35, fallback_floor=0.15):
    """
    Padukan classifier supervised (leksikal, PEMUTUS UTAMA) dengan similarity semantik
    (embeddings, untuk gate out-of-scope + penyelamat saat classifier lemah) untuk
    memutuskan: jawab / klarifikasi / di-luar-topik, plus memori konteks.

    Semua argumen embedding bersifat keyword-only & opsional. Jika embedder None
    (atau centroid tidak tersedia), fungsi jatuh ke jalur TF-IDF+cosine LAMA
    (perilaku persis versi sebelumnya) -- graceful degradation.

    Return: dict {type, tag, confidence, method, candidates, scores, context_note}.
    """
    cleaned = clean_text(user_text)
    vec = build_feature_vector(cleaned, word_vectorizer, char_vectorizer)

    proba = classifier.predict_proba(vec)[0]
    classes = classifier.classes_
    order = proba.argsort()[::-1]
    clf_tag1, p1 = classes[order[0]], float(proba[order[0]])
    clf_tag2, p2 = (classes[order[1]], float(proba[order[1]])) if len(order) > 1 else (None, 0.0)

    embeddings_on = embedder is not None and centroid_matrix is not None and centroid_tags is not None

    # ---- Jalur lama (tanpa embeddings): perilaku persis versi sebelumnya ----
    if not embeddings_on:
        scores = {"clf_top": [(clf_tag1, p1), (clf_tag2, p2)], "sem_top": None}
        if p1 >= threshold:
            return _result("answer", clf_tag1, p1, "classifier", scores=scores)
        sims = cosine_similarity(vec, pattern_vectors)[0]
        idx = sims.argmax()
        if sims[idx] >= fallback_floor:
            return _result("answer", pattern_tags[idx], float(sims[idx]), "cosine_fallback", scores=scores)
        return _result("out_of_scope", OOS_TAG, float(sims[idx]), "cosine_fallback", scores=scores)

    # ---- Jalur hybrid (dengan embeddings) ----
    (sem_tag1, s1), (sem_tag2, s2) = _semantic_scores(user_text, embedder, centroid_matrix, centroid_tags)
    scores = {"clf_top": [(clf_tag1, p1), (clf_tag2, p2)], "sem_top": [(sem_tag1, s1), (sem_tag2, s2)]}

    def clarify(tag_a, score_a, tag_b, score_b):
        cands = [{"tag": tag_a, "score": float(score_a)}, {"tag": tag_b, "score": float(score_b)}]
        return _result("clarify", "", max(score_a, score_b), "clarify", candidates=cands, scores=scores)

    # Gate out-of-scope diperiksa lebih dulu -- tapi follow-up konteks (di bawah) tetap
    # bisa menyelamatkan input pendek "kalau yang premium?" walau di sini kelihatan OOS.
    if _is_out_of_scope(p1, s1):
        result = _result("out_of_scope", OOS_TAG, s1, "oos_gate", scores=scores)
    # R1 -- classifier YAKIN: dipercaya langsung (embedding di sini cuma penanda 'sepakat').
    elif p1 >= t_high:
        method = "hybrid_agree" if sem_tag1 == clf_tag1 else "classifier"
        result = _result("answer", clf_tag1, p1, method, scores=scores)
    # R2 -- classifier SEDANG (t_low..t_high):
    elif p1 >= t_low:
        if sem_tag1 == clf_tag1:
            # semantik menguatkan pilihan classifier
            result = _result("answer", clf_tag1, 0.5 * p1 + 0.5 * s1, "hybrid_agree", scores=scores)
        elif (p1 - p2) < margin_clf:
            # classifier sendiri ragu antara top-2: pakai semantik sbg pemutus bila menunjuk salah satunya,
            # kalau tidak -> klarifikasi.
            if sem_tag1 in (clf_tag1, clf_tag2) and s1 >= s_high:
                result = _result("answer", sem_tag1, 0.5 * p1 + 0.5 * s1, "semantic_rerank", scores=scores)
            else:
                result = clarify(clf_tag1, p1, clf_tag2, p2)
        else:
            # classifier cukup yakin pada top-1 dibanding top-2 -> percaya classifier.
            result = _result("answer", clf_tag1, p1, "classifier", scores=scores)
    # R3 -- classifier LEMAH (p1<t_low) tapi bukan OOS: andalkan semantik.
    else:
        if s1 >= s_high:
            result = _result("answer", sem_tag1, s1, "semantic_fallback", scores=scores)
        else:
            result = clarify(clf_tag1, p1, sem_tag1, s1)

    # ---- Memori konteks: follow-up pendek tanpa subjek mewarisi topik sebelumnya ----
    if _should_carry_context(user_text, prev_context, p1, result["type"]):
        carried = prev_context["last_intent"]
        topic_label = label_for(carried)
        return _result("answer", carried, max(result["confidence"], 0.5), "context_carry",
                       scores=scores, context_note=f"Masih soal {topic_label} ya")

    return result
