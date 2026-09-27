# 🍲 FoodVision AI - Klasifikasi Makanan Lokal (Kikil, Oreg, Kentang)

Aplikasi Web AI berbasis **FastAPI**, **Scikit-Learn / Pickle (.pkl)**, dan **Docker** untuk klasifikasi gambar makanan lokal (**Orek Tempe**, **Olahan Kentang/Balado**, **Kikil Sapi**).

---

## 📁 Struktur Proyek

```text
Klasifikasi Kikil Oreg Kentang/
├── model.pkl             # File model Pickle (.pkl)
├── app.py                # Backend REST API (FastAPI)
├── requirements.txt      # Dependensi minimal
├── Dockerfile            # Container image (Python 3.10-slim)
├── docker-compose.yml    # Konfigurasi Docker
├── README.md             # Panduan
└── templates/
    └── index.html        # Web UI Interaktif
```

---

## 🚀 Cara Menjalankan (Docker)

Pastikan file `model.pkl` sudah berada di folder proyek, lalu jalankan di terminal:

### Perintah Standar:
```bash
docker compose up --build -d
```

### Untuk Menjalankan di Lokal (PowerShell dengan Path Lengkap Docker):
```powershell
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" compose up --build
```

---

## 🌐 Akses Aplikasi

Akses antarmuka web di browser:
👉 **[http://localhost:8000](http://localhost:8000)** *(atau http://127.0.0.1:8000)*

---

## 🛑 Menghentikan Container

```powershell
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" compose down
```
