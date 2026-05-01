from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Dashboard Data Science - Chatbot IPA Kelas 5",
    page_icon="📘",
    layout="wide",
)

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "datasoal.csv"
CLEAN_DIR = BASE_DIR / "data_cleran"
CLEAN_DATA_PATH = CLEAN_DIR / "datasoal_clean.csv"


# =========================================================
# DATA UTILITIES
# =========================================================
def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")

    encodings = ("utf-8-sig", "utf-8", "latin1")
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise last_error


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ÿ\s\-\?\!\,\.\:\;\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = normalize_columns(df)

    # Standarisasi nama kolom yang mungkin berbeda penulisannya
    rename_map = {}
    for col in df.columns:
        normalized = col.replace("_", " ").strip()
        if normalized == "link sumber buku":
            rename_map[col] = "link_sumber_buku"
        elif col in {"link_sumber", "sumber_buku", "link_buku"}:
            rename_map[col] = "link_sumber_buku"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Pastikan kolom inti tersedia
    required_cols = [
        "no",
        "topik",
        "subtopik",
        "soal",
        "jawaban",
        "contoh",
        "konteks",
        "link_sumber_buku",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Isi no kalau kosong / tidak ada
    if "no" not in df.columns or df["no"].isna().all():
        df["no"] = range(1, len(df) + 1)

    text_cols = ["topik", "subtopik", "soal", "jawaban", "contoh", "konteks", "link_sumber_buku"]
    for col in text_cols:
        df[col] = df[col].astype(str).map(clean_text)

    # Hitung panjang teks
    df["soal_len"] = df["soal"].fillna("").map(lambda x: len(str(x).split()))
    df["jawaban_len"] = df["jawaban"].fillna("").map(lambda x: len(str(x).split()))

    # Hapus duplikasi
    before = len(df)
    df = df.drop_duplicates(subset=["soal", "jawaban"]).reset_index(drop=True)
    duplicated_removed = before - len(df)

    # Nomor urut ulang
    df["no"] = range(1, len(df) + 1)

    # Urutan kolom
    preferred = [
        "no",
        "topik",
        "subtopik",
        "soal",
        "jawaban",
        "contoh",
        "konteks",
        "link_sumber_buku",
        "soal_len",
        "jawaban_len",
    ]
    existing = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    return df, duplicated_removed


def save_clean_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, int]:
    raw = read_dataset(RAW_DATA_PATH)
    cleaned, duplicated_removed = prepare_dataset(raw)
    save_clean_dataset(cleaned, CLEAN_DATA_PATH)
    return cleaned, duplicated_removed


@st.cache_resource(show_spinner=False)
def build_index(df: pd.DataFrame):
    corpus = (df["soal"].fillna("") + " " + df["jawaban"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve_answer(query: str, df: pd.DataFrame, vectorizer, matrix):
    query = clean_text(query)
    if not query:
        return None, 0.0

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    best_idx = scores.argmax()
    best_score = float(scores[best_idx])
    best_row = df.iloc[best_idx]
    return best_row, best_score


def top_n_counts(series: pd.Series, n: int = 10, label_name: str = "kategori") -> pd.DataFrame:
    counts = (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .value_counts()
        .head(n)
        .reset_index()
    )
    counts.columns = [label_name, "jumlah"]
    return counts


def plot_bar(ax, data: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str = "Jumlah"):
    ax.barh(data[label_col][::-1], data[value_col][::-1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def calc_missing_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung missing value dengan cara aman:
    - NaN / None dihitung
    - string kosong "" juga dihitung sebagai missing
    """
    temp = df.copy()
    temp = temp.replace(r"^\s*$", pd.NA, regex=True)
    missing = temp.isna().sum().reset_index()
    missing.columns = ["kolom", "missing"]
    return missing


# =========================================================
# UI
# =========================================================
def main():
    st.title("📘 Dashboard Data Science – Chatbot Edukatif IPA Kelas 5")
    st.caption(
        "Dashboard ini menampilkan data wrangling, EDA, dan demo retrieval sederhana "
        "untuk dataset Q&A IPA SD."
    )

    try:
        df, duplicated_removed = load_data()
    except Exception as exc:
        st.error(f"Gagal memuat dataset: {exc}")
        st.stop()

    vectorizer, matrix = build_index(df)

    # Sidebar filter
    st.sidebar.header("Filter Data")
    topics = ["Semua"] + sorted(
        [t for t in df["topik"].dropna().astype(str).unique().tolist() if t.strip()]
    )
    selected_topic = st.sidebar.selectbox("Pilih topik", topics)

    if selected_topic != "Semua":
        show_df = df[df["topik"] == selected_topic].copy()
    else:
        show_df = df.copy()

    st.sidebar.write(f"Jumlah data terfilter: **{len(show_df)}**")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total baris", f"{len(df):,}")
    c2.metric("Jumlah topik", f"{df['topik'].nunique():,}")
    c3.metric("Jumlah subtopik", f"{df['subtopik'].nunique():,}")
    c4.metric("Data duplikat terhapus", f"{duplicated_removed:,}")

    # Section 1
    st.subheader("1) Ringkasan Kualitas Data")

    q1, q2 = st.columns(2)

    with q1:
        st.write("Kolom dataset")
        col_info = pd.DataFrame(
            {
                "kolom": df.columns,
                "tipe": [str(t) for t in df.dtypes],
            }
        )
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    with q2:
        missing = calc_missing_counts(df)
        missing_nonzero = missing[missing["missing"] > 0].sort_values("missing", ascending=True)

        if missing_nonzero.empty:
            st.success("Tidak ada missing value pada dataset.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_bar(
                ax,
                missing_nonzero.tail(8),
                "kolom",
                "missing",
                "Missing Value per Kolom",
                xlabel="Jumlah missing",
            )
            st.pyplot(fig, clear_figure=True)

    # Section 2
    st.subheader("2) EDA Dataset")

    a1, a2 = st.columns(2)

    with a1:
        top_topics = top_n_counts(df["topik"], n=10, label_name="topik")
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_bar(ax, top_topics, "topik", "jumlah", "10 Topik Terbanyak")
        st.pyplot(fig, clear_figure=True)

    with a2:
        top_subtopics = top_n_counts(df["subtopik"], n=10, label_name="subtopik")
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_bar(ax, top_subtopics, "subtopik", "jumlah", "10 Subtopik Terbanyak")
        st.pyplot(fig, clear_figure=True)

    b1, b2 = st.columns(2)

    with b1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["soal_len"], bins=20)
        ax.set_title("Distribusi Panjang Soal")
        ax.set_xlabel("Jumlah kata")
        ax.set_ylabel("Frekuensi")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    with b2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["jawaban_len"], bins=20)
        ax.set_title("Distribusi Panjang Jawaban")
        ax.set_xlabel("Jumlah kata")
        ax.set_ylabel("Frekuensi")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    # Section 3
    st.subheader("3) Contoh Data")
    st.dataframe(
        show_df[["no", "topik", "subtopik", "soal", "jawaban"]].head(20),
        use_container_width=True,
        hide_index=True,
    )

    # Section 4
    st.subheader("4) Demo Pencarian Jawaban Sederhana")
    query = st.text_input("Ketik pertanyaan IPA siswa, lalu sistem mencari jawaban paling mirip:")

    if query:
        best_row, score = retrieve_answer(query, df, vectorizer, matrix)
        if best_row is None:
            st.warning("Masukkan pertanyaan terlebih dahulu.")
        else:
            st.success(f"Skor kemiripan: {score:.2f}")
            st.write(f"**Topik:** {best_row['topik']}")
            st.write(f"**Subtopik:** {best_row['subtopik']}")
            st.write(f"**Soal:** {best_row['soal']}")
            st.write(f"**Jawaban:** {best_row['jawaban']}")

            if "contoh" in best_row.index:
                st.write(f"**Contoh:** {best_row['contoh']}")
            if "konteks" in best_row.index:
                st.write(f"**Konteks:** {best_row['konteks']}")

    # Section 5
    st.subheader("5) File Data Bersih")
    st.info(f"Dataset bersih otomatis disimpan ke: {CLEAN_DATA_PATH}")

    with st.expander("Lihat 10 baris data bersih"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()