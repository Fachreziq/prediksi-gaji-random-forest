from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)

# Model Random Forest hasil notebook TA-06
model = joblib.load(BASE / "model_rf_gaji.pkl")

# Ganti kalau satuan gaji di dataset berbeda (cek deskripsi dataset di Kaggle)
MATA_UANG = "$"


def format_angka(n):
    """71499.27 -> '71.499' (pemisah ribuan titik, gaya Indonesia)."""
    return f"{n:,.0f}".replace(",", ".")


def muat_titik():
    """Baca dataset untuk digambar di grafik. Kalau gagal, grafik tampil tanpa titik data."""
    try:
        df = pd.read_csv(BASE / "Salary_dataset.csv")
        df["YearsExperience"] = df["YearsExperience"].round(1)
        return [{"x": float(x), "y": float(y)}
                for x, y in zip(df["YearsExperience"], df["Salary"])]
    except Exception:
        return []


TITIK = muat_titik()
if TITIK:
    MIN_DATA = min(t["x"] for t in TITIK)
    MAX_DATA = max(t["x"] for t in TITIK)
else:
    MIN_DATA, MAX_DATA = 1.2, 10.6

# Kurva prediksi model (berbentuk tangga) untuk digambar di grafik
_grid = pd.DataFrame({"YearsExperience": np.linspace(MIN_DATA, MAX_DATA, 200)})
KURVA = [{"x": float(x), "y": float(y)}
         for x, y in zip(_grid["YearsExperience"], model.predict(_grid))]

N_POHON = int(model.n_estimators)
KEDALAMAN = model.max_depth if model.max_depth is not None else "tanpa batas"


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    error = None
    peringatan = None
    pengalaman = ""
    pengalaman_angka = None

    if request.method == "POST":
        pengalaman = request.form.get("pengalaman", "").strip()
        try:
            nilai = float(pengalaman.replace(",", "."))
            if nilai < 0 or nilai > 50:
                error = "Masukkan pengalaman antara 0 dan 50 tahun."
            else:
                data = pd.DataFrame({"YearsExperience": [nilai]})
                hasil = float(model.predict(data)[0])
                pengalaman_angka = nilai
                if nilai < MIN_DATA or nilai > MAX_DATA:
                    peringatan = (
                        f"Data latih hanya mencakup {MIN_DATA:g}-{MAX_DATA:g} tahun. "
                        "Random Forest tidak bisa memperkirakan di luar rentang itu: "
                        "hasilnya sama dengan nilai di batas data latih terdekat."
                    )
        except ValueError:
            error = "Masukkan angka yang valid, contoh: 5 atau 3,5."

    return render_template(
        "index.html",
        hasil=hasil,
        hasil_teks=format_angka(hasil) if hasil is not None else None,
        error=error,
        peringatan=peringatan,
        pengalaman=pengalaman,
        pengalaman_angka=pengalaman_angka,
        titik=TITIK,
        kurva=KURVA,
        mata_uang=MATA_UANG,
        jumlah_data=len(TITIK),
        n_pohon=N_POHON,
        kedalaman=KEDALAMAN,
    )


if __name__ == "__main__":
    app.run(debug=True)