"""
Dashboard Chatbot "Toko Sembako Barokah" -- UAS Data Mining/Chatbot/NLP.
Modern glassmorphic UI with fluid interactions.
Jalankan dengan: streamlit run app.py
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

import chatbot_engine as engine
from train_model import (
    evaluate_context_cases,
    evaluate_oos_cases,
    evaluate_paraphrase_cases,
    evaluate_slang_typo_cases,
    evaluate_smalltalk_cases,
    train_and_evaluate,
)
from ui_helpers import (
    divider,
    info_card,
    meta_badge,
    render_test_section,
    result_card,
    section_header,
    subtitle,
    success_banner,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXIT_KEYWORDS = {"exit", "keluar", "selesai", "bye", "quit", "stop"}
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Toko Sembako Barokah - ChatBot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load external CSS & js
# ---------------------------------------------------------------------------
_css = (BASE_DIR / 'style.css').read_text(encoding='utf-8')
st.html(f"<style>{_css}</style>")

_js = (BASE_DIR / 'scripts.js').read_text(encoding='utf-8')
st.html(f"<script>{_js}</script>", unsafe_allow_javascript=True)

# Google Font (loaded via <link> since @import doesn't work in Streamlit)
st.html(
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap">'
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model chatbot...")
def get_artifacts():
    return train_and_evaluate()


artifacts = get_artifacts()
dataset = artifacts["dataset"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🛒 Toko Sembako Barokah")
st.markdown(
    f"""
    <div style='color: #6B7280; font-size: 1rem; margin-bottom: 2rem;'>
        UAS Data Mining / Chatbot / NLP — <strong>Muhamad Abdul Aziz</strong> —
        Tema: Penjualan Sembako
        <span style='color: #7C3AED; font-weight: 600;'>({artifacts['n_intents']} intent, {artifacts['n_samples']} pattern latih)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_chat, tab_model = st.tabs(["💬 Chat Asisten", "📊 Model & Akurasi"])


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------
def _reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Selamat datang di Toko Sembako Barokah. "
                                         "Ada yang bisa kami bantu? (ketik 'keluar' untuk mengakhiri)"}
    ]
    st.session_state.context = None
    st.session_state.pending_clarify = None
    st.session_state.ended = False


def _typing_effect(text):
    """Generator that yields text word-by-word with natural typing speed."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.02 + min(len(word) * 0.004, 0.04))


def _emit_answer(tag, user_input, result):
    """Tambahkan jawaban bot ke riwayat + update memori konteks."""
    response = engine.get_response(tag, dataset)
    if result.get("context_note"):
        response = f"_{result['context_note']} —_ {response}"
    meta = f"intent: {tag} | confidence: {result['confidence']:.2f} | metode: {result['method']}"
    st.session_state.messages.append({"role": "assistant", "content": response, "meta": meta})
    st.session_state.context = engine.update_context(
        st.session_state.context, {"type": "answer", "tag": tag}, user_input
    )


# ---------------------------------------------------------------------------
# TAB 1: CHAT
# ---------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        _reset_chat()

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### 🎛️ Kontrol Percakapan")
        divider()

        if st.button("🔄 Reset Percakapan", use_container_width=True, type="primary"):
            _reset_chat()
            st.rerun()

        divider()
        info_card(
            "💡 Tips",
            "Ketik salah satu kata berikut untuk mengakhiri chat:<br>"
            "<code>keluar</code> <code>selesai</code> <code>bye</code> <code>exit</code>",
            variant="tip",
        )

        emb_ok = artifacts.get("embedding_available")
        info_card(
            "🧠 Pemahaman Semantik",
            f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<div style='width:10px;height:10px;border-radius:50%;"
            f"background:{'#10B981' if emb_ok else '#F59E0B'};"
            f"box-shadow:0 0 8px {'#10B981' if emb_ok else '#F59E0B'};'></div>"
            f"<span>{'Aktif' if emb_ok else 'Nonaktif (Mode TF-IDF)'}</span></div>",
            variant="info" if emb_ok else "warning",
        )

        with st.expander("📋 Lihat Semua Intent Dataset"):
            for tag in sorted(i["tag"] for i in dataset):
                st.markdown(f"• `{tag}`")

    # ---- Chat messages ----
    for idx, msg in enumerate(st.session_state.messages):
        is_streaming = (
            idx == len(st.session_state.messages) - 1
            and msg["role"] == "assistant"
            and st.session_state.get("_just_streamed", False)
        )

        with st.chat_message(msg["role"]):
            if is_streaming:
                st.write_stream(_typing_effect(msg["content"]))
            else:
                st.markdown(msg["content"])
            if "meta" in msg:
                meta_badge(msg["meta"])

    # ---- Clarification buttons ----
    if st.session_state.get("pending_clarify") and not st.session_state.ended:
        cands = st.session_state.pending_clarify
        divider()
        st.markdown("**Pilih salah satu opsi:**")
        for i, (col, cand) in enumerate(zip(st.columns(len(cands)), cands)):
            with col:
                if st.button(engine.label_for(cand["tag"]), key=f"clarify_{i}",
                             use_container_width=True, type="primary"):
                    label = engine.label_for(cand["tag"])
                    st.session_state.messages.append({"role": "user", "content": label})
                    _emit_answer(cand["tag"], label,
                                 {"type": "answer", "tag": cand["tag"],
                                  "confidence": cand["score"], "method": "clarify_pilih",
                                  "context_note": None})
                    st.session_state.pending_clarify = None
                    st.rerun()

    # ---- Ended / Input ----
    if st.session_state.ended:
        info_card(
            "✓ Percakapan Telah Diakhiri",
            "Klik tombol <strong>Reset Percakapan</strong> di sidebar untuk memulai kembali.",
            variant="tip",
        )
    else:
        user_input = st.chat_input("Tulis pertanyaan Anda di sini...")
        if user_input:
            st.session_state.pending_clarify = None
            st.session_state._just_streamed = False
            st.session_state.messages.append({"role": "user", "content": user_input})

            if user_input.strip().lower() in EXIT_KEYWORDS:
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": "Terima kasih sudah mengunjungi Toko Sembako Barokah. Sampai jumpa!"}
                )
                st.session_state.ended = True
                st.session_state._just_streamed = True
            else:
                with st.expander("Hasil preprocessing (debug)", expanded=False):
                    st.write({
                        "token": engine.tokenize(user_input),
                        "teks_leksikal (TF-IDF)": engine.clean_text(user_input),
                        "teks_semantik (embedding)": engine.clean_text_semantic(user_input),
                    })

                result = engine.predict_intent(
                    user_input,
                    artifacts["word_vectorizer"],
                    artifacts["char_vectorizer"],
                    artifacts["classifier"],
                    artifacts["pattern_vectors"],
                    artifacts["pattern_tags"],
                    embedder=artifacts.get("embedder"),
                    centroid_matrix=artifacts.get("centroid_matrix"),
                    centroid_tags=artifacts.get("centroid_tags"),
                    prev_context=st.session_state.context,
                )

                if result["type"] == "answer":
                    _emit_answer(result["tag"], user_input, result)
                elif result["type"] == "clarify":
                    st.session_state.messages.append(
                        {"role": "assistant", "content": engine.build_clarify_text(result["candidates"]),
                         "meta": "metode: clarify"}
                    )
                    st.session_state.pending_clarify = result["candidates"]
                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": engine.OOS_RESPONSE, "meta": "metode: oos_gate"}
                    )

                st.session_state._just_streamed = True
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: MODEL & AKURASI
# ---------------------------------------------------------------------------
with tab_model:
    section_header("📊", "Ringkasan Model")
    divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Intent", artifacts["n_intents"])
    col2.metric("Akurasi (Test Split 25%)", f"{artifacts['accuracy']:.1%}")
    col3.metric("Akurasi (5-Fold CV)", f"{artifacts['cv_mean']:.1%}", f"± {artifacts['cv_std']:.1%}")

    divider()
    info_card(
        "ℹ️ Catatan Metodologi",
        f"Dataset ini relatif kecil (24 intent, {artifacts['n_samples']} sample pattern), "
        "sehingga akurasi dari satu kali train/test split bisa bervariasi. Karena itu akurasi "
        "cross-validation 5-fold ditampilkan berdampingan sebagai estimasi yang lebih stabil, "
        "bukan memilih angka yang lebih bagus saja.",
    )

    # ---- Classifier comparison ----
    section_header("🔬", "Perbandingan Algoritma Classifier")
    subtitle(
        "Beberapa algoritma supervised learning dibandingkan lewat 5-fold cross-validation "
        "pada fitur TF-IDF gabungan (kata + karakter). Model dengan skor rata-rata tertinggi "
        "dipilih otomatis sebagai model produksi yang dipakai chat."
    )

    comparison_df = pd.DataFrame(artifacts["classifier_comparison"])
    st.dataframe(
        comparison_df.style.format({"cv_mean": "{:.2%}", "cv_std": "{:.2%}"}),
        use_container_width=True,
    )
    success_banner(f"✓ Model Terpenuhi (Produksi): {artifacts['best_model_name']}")

    # ---- Embedding comparison ----
    if artifacts.get("embedding_available") and artifacts.get("embedding_comparison"):
        section_header("🧠", "Perbandingan TF-IDF vs Embedding")
        subtitle(
            "Selain classifier TF-IDF di atas, dievaluasi juga classifier berbasis "
            "sentence-embeddings (NearestCentroid & kNN cosine) pada fold yang sama. "
            "Classifier produksi tetap yang TF-IDF."
        )
        emb_df = pd.DataFrame(artifacts["embedding_comparison"])
        st.dataframe(
            emb_df.style.format({"cv_mean": "{:.2%}", "cv_std": "{:.2%}"}),
            use_container_width=True,
        )
    else:
        info_card("⚠️ Layer Embedding Nonaktif",
                  "Model tidak termuat — chatbot berjalan mode TF-IDF saja.", variant="warning")

    # ---- Confusion matrix & classification report ----
    divider()
    section_header("📈", "Confusion Matrix")
    subtitle("Pada data test split")
    cm_df = pd.DataFrame(artifacts["confusion_matrix"], index=artifacts["labels"], columns=artifacts["labels"])
    st.dataframe(cm_df, use_container_width=True)

    section_header("📝", "Classification Report")
    subtitle("Pada data test split")
    report_df = pd.DataFrame(artifacts["classification_report"]).transpose()
    st.dataframe(
        report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}),
        use_container_width=True,
    )

    # ---- Evaluation tests (using reusable render_test_section) ----
    divider()

    render_test_section(
        "🔄", "Uji Paraphrase",
        "Membuktikan requirement 'chatbot harus mengenali kemiripan input user, tidak harus "
        "persis 100% sama dengan data latih'. Beberapa kasus punya lebih dari satu tag yang "
        "dianggap benar karena maknanya memang berdekatan.",
        pd.DataFrame(evaluate_paraphrase_cases(artifacts)),
    )

    render_test_section(
        "💬", "Uji Slang & Typo",
        "Menguji chatbot dengan kalimat bergaya chat asli: singkatan/bahasa gaul (gmn, brp, "
        "gw, ga), typo umum, dan kalimat santai tanpa tanda baca.",
        pd.DataFrame(evaluate_slang_typo_cases(artifacts)),
    )

    if artifacts.get("embedding_available"):
        divider()

        render_test_section(
            "🔗", "Uji Memori Konteks",
            "Menguji apakah bot mengingat topik giliran sebelumnya saat pengguna bertanya "
            "singkat tanpa subjek (mis. 'kalau yang premium?').",
            pd.DataFrame(evaluate_context_cases(artifacts)),
        )

        render_test_section(
            "🚫", "Uji Out-of-Scope",
            "Pertanyaan di luar topik toko harus dijawab dengan pengarahan sopan "
            "(bukan dipaksa ke intent terdekat).",
            pd.DataFrame(evaluate_oos_cases(artifacts)),
        )

        render_test_section(
            "👋", "Uji Basa-basi (Small-talk)",
            "Sapaan & pertanyaan ringan (kamu siapa, apa kabar, bisa bantu apa) dijawab wajar.",
            pd.DataFrame(evaluate_smalltalk_cases(artifacts)),
        )
