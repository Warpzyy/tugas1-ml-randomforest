# Simulasi Random Forest — Prediksi Telat Mengumpulkan Tugas

Web simulasi Random Forest (Gain Ratio) berbasis Flask: bootstrap sampling, pemilihan fitur acak (m), entropy, gain, split info, gain ratio, pohon keputusan, dan majority vote.

## Menjalankan lokal
```
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:5000

## Deploy ke Vercel
1. Push repo ini ke GitHub.
2. Di vercel.com pilih **Add New → Project**, import repo ini (Framework: Flask terdeteksi otomatis dari `app.py`).
3. Klik **Deploy**, lalu salin link-nya ke laporan.

Struktur: `app.py` (route), `rf.py` (algoritma), `templates/index.html` (antarmuka).
