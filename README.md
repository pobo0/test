# Submission – Chatbot Edukatif IPA Kelas 5

Struktur project:

```text
submission/
├── dashboard/
│   ├── dashboard.py
│   └── data_cleran/
├── data/
│   └── datasoal.csv
├── notebook.ipynb
├── requirements.txt
├── README.md
└── url.txt
```

## 1) Instalasi

Buat virtual environment lalu install dependency:

```bash
pip install -r requirements.txt
```

## 2) Menjalankan dashboard Streamlit

Jalankan dari folder `submission`:

```bash
streamlit run dashboard/dashboard.py
```

Dashboard akan:
- membaca dataset dari `data/datasoal.csv`
- melakukan cleaning otomatis
- menyimpan data bersih ke `dashboard/data_cleran/datasoal_clean.csv`
- menampilkan EDA dan demo pencarian jawaban sederhana

## 3) Menjalankan notebook

Buka notebook:

```bash
jupyter notebook notebook.ipynb
```

Notebook berisi:
- loading dataset
- data cleaning
- EDA
- visualisasi
- baseline retrieval sederhana

## 4) Deploy ke Streamlit Community Cloud

1. Upload folder `submission` ke GitHub.
2. Pastikan file utama app adalah `dashboard/dashboard.py`.
3. Tambahkan dependency di `requirements.txt`.
4. Saat membuat app di Streamlit Cloud, isi path file utama dengan:
   `dashboard/dashboard.py`

## 5) url.txt

Isi `url.txt` dengan link deployment setelah aplikasi berhasil dipublish.

Contoh:

```text
https://nama-app-kamu.streamlit.app
```

## 6) Catatan

Folder `data_cleran` dipakai untuk menyimpan dataset yang sudah dibersihkan oleh dashboard.
