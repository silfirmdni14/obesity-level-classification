import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config (
    page_title="HealthPredict",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

rf_model = joblib.load("model_klasifikasi_prediksi_tingkat_obesitas.joblib")

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

#TEMA
NAVY       = "#2c3e7a"
NAVY_DARK  = "#1c2a55"
TEAL       = "#1D9E75"
TEAL_LIGHT = "#27c994"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Background app */
.stApp {{
    background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 100%);
}}

/* Container utama */
.block-container {{
    max-width: 1200px;
    padding-top: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}}

/* HERO */
.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 60%, #0f1a3a 100%);
    border-radius: 22px;
    padding: 38px 44px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(44,62,122,0.25);
    margin-bottom: 24px;
}}
.hero::before {{
    content: "";
    position: absolute; top:-90px; right:-90px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, {TEAL}55 0%, transparent 70%);
    border-radius: 50%;
}}
.hero h1 {{ font-size: 38px; font-weight: 800; margin: 6px 0 8px 0; letter-spacing: -0.5px; }}
.hero p  {{ font-size: 16px; opacity: 0.9; max-width: 720px; line-height: 1.6; margin:0; }}
.hero .badge {{
    display: inline-block;
    background: rgba(29,158,117,0.2);
    border: 1px solid {TEAL};
    color: {TEAL_LIGHT};
    padding: 5px 14px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DARK} 100%);
}}
section[data-testid="stSidebar"] * {{ color: #ffffff !important; }}

.sidebar-logo {{
    text-align: center;
    padding: 18px 10px 6px 10px;
}}
.sidebar-logo .logo-circle {{
    width: 68px; height: 68px;
    margin: 0 auto 10px auto;
    border-radius: 50%;
    background: linear-gradient(135deg, {TEAL} 0%, {TEAL_LIGHT} 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px;
    box-shadow: 0 8px 22px rgba(29,158,117,0.45);
}}
.sidebar-logo h2 {{ margin: 0; font-weight: 800; letter-spacing: 0.5px; }}
.sidebar-logo p  {{ margin: 2px 0 0 0; font-size: 12px; opacity: 0.8; }}

.guide-card {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 16px;
    font-size: 13px;
    line-height: 1.6;
}}
.guide-card h3 {{
    font-size: 14px; margin: 0 0 8px 0; color: {TEAL_LIGHT} !important;
    text-transform: uppercase; letter-spacing: 1px;
}}
.guide-card ol, .guide-card ul {{ margin: 0; padding-left: 18px; }}
.guide-card li {{ margin-bottom: 4px; }}

/* TABS */
.stTabs {{
    margin-top: 1rem;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: white;
    padding: 8px;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(44,62,122,0.06);
    overflow: visible !
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    color: #6b7a99;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {NAVY}, {NAVY_DARK}) !important;
    color: white !important;
}}

/* CARDS */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: white;
    border-radius: 16px !important;
    border: 1px solid #e6ecf3 !important;
    box-shadow: 0 4px 18px rgba(44,62,122,0.06);
    transition: all 0.25s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 10px 28px rgba(44,62,122,0.12);
    transform: translateY(-2px);
}}

/* Metric cards (custom html) */
.card {{
    padding: 22px;
    border: 1px solid #e6ecf3;
    border-radius: 16px;
    text-align: center;
    background: white;
    box-shadow: 0 4px 18px rgba(44,62,122,0.06);
    transition: all 0.25s ease;
}}
.card:hover {{ transform: translateY(-3px); box-shadow: 0 12px 28px rgba(44,62,122,0.12); }}
.card h3 {{ font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; color: {NAVY}; }}
.card p  {{ font-size: 0.85rem; color: #6b7a99; margin: 0; text-transform: uppercase; letter-spacing: 1px; }}

.card-large {{
    padding: 28px;
    border: 1px solid #e6ecf3;
    border-radius: 18px;
    background: white;
    box-shadow: 0 4px 20px rgba(44,62,122,0.06);
    margin-top: 16px;
}}
.card-large h3 {{
    font-size: 1.15rem; font-weight: 700; color: {NAVY};
    margin: 0 0 12px 0; display:flex; align-items:center; gap:10px;
}}
.card-large h3::before {{
    content:""; width:8px; height:8px; border-radius:50%;
    background: {TEAL}; box-shadow: 0 0 0 4px rgba(29,158,117,0.2);
}}
.card-large p {{ color: #4a5573; line-height: 1.75; margin-bottom: 8px; }}

.card-metric {{
    background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
    border: 1px solid #e6ecf3;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(44,62,122,0.06);
}}
.card-metric p  {{ color: #6b7a99; font-size: 0.78rem; margin-bottom: 6px; text-transform:uppercase; letter-spacing:1.2px; }}
.card-metric h2 {{ color: {NAVY}; margin: 0; font-size: 2rem; font-weight: 800; }}

.card-section {{
    background: white;
    border: 1px solid #e6ecf3;
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(44,62,122,0.06);
}}
.card-section h3 {{ color: {NAVY}; margin-top: 0; font-weight: 700; }}
.card-section ol, .card-section ul {{ color: #4a5573; line-height: 1.9; }}
.card-section p {{ color: {NAVY}; font-weight: 600; }}

/* Tabel istilah */
.tabel-istilah {{ width:100%; border-collapse: collapse; }}
.tabel-istilah td {{ padding: 8px 12px; color: #4a5573; border-bottom: 1px solid #f0f3f8; }}
.tabel-istilah td:first-child {{ font-weight: 600; color: {NAVY}; white-space: nowrap; }}

/* Card chart */
.card-chart {{
    border: 1px solid #e6ecf3;
    border-radius: 16px;
    padding: 18px;
    background: white;
    box-shadow: 0 4px 18px rgba(44,62,122,0.06);
    margin-bottom: 16px;
    transition: all 0.25s ease;
}}
.card-chart:hover {{ box-shadow: 0 10px 28px rgba(44,62,122,0.10); }}

/*  BUTTON */
.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, {TEAL} 0%, {TEAL_LIGHT} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 0.3px;
    box-shadow: 0 8px 22px rgba(29,158,117,0.35);
    transition: all 0.25s ease;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(29,158,117,0.5);
    filter: brightness(1.05);
}}

/* INPUTS  */
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{
    border-radius: 10px !important;
}}

/* Section title helper */
.section-title {{
    display:flex; align-items:center; gap:10px;
    font-size: 18px; font-weight: 700; color: {NAVY};
    margin: 4px 0 14px 0;
}}
.section-title .dot {{
    width:10px; height:10px; border-radius:50%;
    background: {TEAL};
    box-shadow: 0 0 0 4px rgba(29,158,117,0.2);
}}

/* Tip card */
.tip-card {{
    background: white;
    border-left: 4px solid {TEAL};
    padding: 12px 16px;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(44,62,122,0.06);
    margin-bottom: 10px;
    color: #2a3148;
    font-size: 0.92rem;
}}

/* About / profile */
.profile-card {{
    background: linear-gradient(135deg, white 0%, #f5f9ff 100%);
    border-radius: 22px; padding: 30px;
    text-align: center;
    border: 1px solid #e6ecf3;
    box-shadow: 0 10px 30px rgba(44,62,122,0.08);
}}
.profile-avatar {{
    width: 130px; height: 130px;
    margin: 0 auto 14px auto;
    border-radius: 50%;
    background: linear-gradient(135deg, {NAVY}, {TEAL});
    display:flex; align-items:center; justify-content:center;
    color: white; font-size: 56px; font-weight: 800;
    box-shadow: 0 10px 28px rgba(44,62,122,0.3);
}}
.skill-pill {{
    display:inline-block; padding: 6px 14px; margin: 4px;
    background: {NAVY}; color: white !important;
    border-radius: 999px; font-size: 12px; font-weight: 600;
}}
.skill-pill.teal {{ background: {TEAL}; }}
.social-btn {{
    display:inline-block; padding: 10px 18px; margin: 4px;
    background: white; color: {NAVY} !important;
    border: 1.5px solid {NAVY};
    border-radius: 12px; font-weight: 600; text-decoration: none;
    transition: all 0.2s;
}}
.social-btn:hover {{ background: {NAVY}; color: white !important; }}
</style>
""", unsafe_allow_html=True)


# SIDEBAR — Logo + Panduan
with st.sidebar:
    # Logo & nama aplikasi
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-circle">🩺</div>
        <h2>HealthPredict</h2>
    </div>
    """, unsafe_allow_html=True)
    # Panduan penggunaan
    st.markdown("""
    <div class="guide-card">
        <h3>📘 Cara Menggunakan</h3>
        <ol>
            <li>Pilih tab <b>Prediksi Obesitas</b></li>
            <li>Isi semua form sesuai kondisi diri</li>
            <li>Klik tombol <b>Prediksi</b></li>
            <li>Lihat hasil & rekomendasi kesehatan</li>
        </ol>
    </div>

    <div class="guide-card">
        <h3>✅ Petunjuk Pengisian</h3>
        <ul>
            <li>Isi data dengan jujur</li>
            <li>Jangan kosongkan field apapun</li>
            <li>Periksa kembali sebelum prediksi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3,tab4 = st.tabs([ "📥  Prediksi Obesitas", "📊  Informasi",  "📈 Analisis Dataset", "👤    Tentang Saya"])

# TAB 1 — PREDIKSI
with tab1:
    st.markdown('<div class="section-title"><span class="dot"></span>Form Data Kesehatan</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.markdown("#### 🧍 Data Diri")
            # ── agar selectbox wajib dipilih
            gender = st.selectbox("Jenis Kelamin",options=["Laki-laki", "Perempuan"],index=None,placeholder="Pilih jenis kelamin")
            gender_model = "Male" if gender == "Laki-laki" else "Female" if gender == "Perempuan" else None
            age = st.number_input("Umur",value=None,step=1,placeholder="Masukkan umur Anda")
            height_cm = st.number_input("Tinggi Badan (cm)",value=None,step=1,placeholder="Contoh: 165")
            height = (height_cm / 100) if height_cm else None
            weight = st.number_input("Berat Badan (kg)",value=None,step=1,placeholder="Contoh: 60")
            family_history = st.selectbox("Riwayat Obesitas Keluarga",options=["Ya", "Tidak"],index=None,placeholder="Pilih riwayat keluarga...")
            family_history_model = "yes" if family_history == "Ya" else "no" if family_history == "Tidak" else None

    with col_b:
        with st.container(border=True):
            st.markdown("#### 🍽️ Kebiasaan Makan")

            favc = st.selectbox("Sering Makan Makanan Tinggi Kalori",options=["Ya", "Tidak"],index=None,placeholder="Pilih kebiasaan makan...")
            favc_model = "yes" if favc == "Ya" else "no" if favc == "Tidak" else None
            caec = st.selectbox("Kebiasaan Ngemil",options=["Tidak", "Kadang-kadang", "Sering", "Selalu"],index=None,placeholder="Pilih frekuensi ngemil...")
            caec_model = ("no" if caec == "Tidak" else "Sometimes"  if caec == "Kadang-kadang" else"Frequently" if caec == "Sering" else"Always" if caec == "Selalu" else None)
            st.caption("1 = jarang, 2 = cukup sering, 3 = sangat sering")
            fcvc = st.slider("Frekuensi Konsumsi Sayur", min_value=1, max_value=3, value=None)
            ncp  = st.slider("Jumlah Makan per hari",    min_value=1, max_value=4, value=None)
            st.caption("1 = <1 liter, 2 = 1–2 liter, 3 = >2 liter")
            ch2o = st.slider("Konsumsi Air putih per hari", min_value=1, max_value=3, value=None)

    col_c, col_d = st.columns(2)

    with col_c:
        with st.container(border=True):
            st.markdown("#### 🏃 Aktivitas Fisik")
            st.caption("1 = 1–2 kali/minggu • 2 = 3–4 kali/minggu • 3 = rutin hampir setiap hari")
            faf = st.slider("Frekuensi Olahraga per Minggu",  min_value=1, max_value=3, value=None)
            st.caption("1 = 1–3 jam, 2 = 4–6 jam/hari, 3 = lebih dari 6 jam/hari")
            tue = st.slider("Waktu penggunaan Gadget per hari", min_value=1, max_value=3, value=None)
            mtrans = st.selectbox("Transportasi",options=["Jalan Kaki", "Sepeda", "Motor", "Kendaraan Umum", "Mobil"],index=None,placeholder="Pilih jenis transportasi...")
            mtrans_model = ("Walking" if mtrans == "Jalan Kaki" else"Bike" if mtrans == "Sepeda" else"Motorbike" if mtrans == "Motor" else"Public_Transportation" if mtrans == "Kendaraan Umum"  else"Car" if mtrans == "Mobil" else None)

    with col_d:
        with st.container(border=True):
            st.markdown("#### 🌿 Gaya Hidup")
            calc = st.selectbox("Konsumsi Alkohol",options=["Tidak", "Kadang-kadang", "Sering", "Selalu"],index=None,placeholder="Pilih frekuensi konsumsi alkohol...")
            calc_model = ("no" if calc == "Tidak" else"Sometimes"  if calc == "Kadang-kadang" else"Frequently" if calc == "Sering" else"Always"  if calc == "Selalu"  else None)
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("ℹ️ Pastikan semua data telah terisi dengan benar sebelum menekan tombol prediksi.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Prediksi Tingkat Obesitas Saya", type="primary")

    if predict_clicked:
        # Selectbox — None berarti user belum menyentuhnya sama sekali
        errors = []
        if gender       is None: errors.append("⚠️ **Jenis Kelamin** belum dipilih.")
        if family_history is None: errors.append("⚠️ **Riwayat Obesitas Keluarga** belum dipilih.")
        if favc         is None: errors.append("⚠️ **Sering Makan Makanan Tinggi Kalori** belum dipilih.")
        if caec         is None: errors.append("⚠️ **Kebiasaan Ngemil** belum dipilih.")
        if mtrans       is None: errors.append("⚠️ **Transportasi** belum dipilih.")
        if calc         is None: errors.append("⚠️ **Konsumsi Alkohol** belum dipilih.")

        # Number input — None berarti benar-benar kosong; 0 juga tidak masuk akal
        if age is None or age <= 0:
            errors.append("⚠️ **Umur** belum diisi atau tidak valid (harus > 0).")
        elif age < 5 or age > 120:
            errors.append("⚠️ **Umur** harus berada di antara 5–120 tahun.")
        if height_cm is None or height_cm <= 0:
            errors.append("⚠️ **Tinggi Badan** belum diisi atau tidak valid (harus > 0).")
        elif height_cm < 50 or height_cm > 250:
            errors.append("⚠️ **Tinggi Badan** harus berada di antara 50–250 cm.")
        if weight is None or weight <= 0:
            errors.append("⚠️ **Berat Badan** belum diisi atau tidak valid (harus > 0).")
        elif weight < 10 or weight > 300:
            errors.append("⚠️ **Berat Badan** harus berada di antara 10–300 kg.")

        # Slider — value=None berarti user belum menggesernya
        if fcvc  is None: errors.append("⚠️ **Frekuensi Konsumsi Sayur** belum dipilih.")
        if ncp   is None: errors.append("⚠️ **Jumlah Makan per hari** belum dipilih.")
        if ch2o  is None: errors.append("⚠️ **Konsumsi Air putih per hari** belum dipilih.")
        if faf   is None: errors.append("⚠️ **Frekuensi Olahraga per Minggu** belum dipilih.")
        if tue   is None: errors.append("⚠️ **Waktu penggunaan Gadget per hari** belum dipilih.")

        # Tampilkan semua error sekaligus; hentikan eksekusi
        if errors:
            st.error("🚫 **Harap lengkapi semua field berikut sebelum melakukan prediksi:**")
            for msg in errors:
                st.warning(msg)
        st.stop()
        data_baru = pd.DataFrame(
            [[gender_model, age, height, weight, family_history_model, favc_model,
              fcvc, ncp, caec_model, ch2o, faf, tue, calc_model, mtrans_model]],
            columns=["Gender", "Age", "Height", "Weight", "family_history_with_overweight",
                     "FAVC", "FCVC", "NCP", "CAEC", "CH2O", "FAF", "TUE", "CALC", "MTRANS"]
        )
        data_baru["BMI"]           = data_baru["Weight"] / (data_baru["Height"] ** 2)
        data_baru["BMI_Category"]  = data_baru["BMI"].apply(bmi_category)
        data_baru["ActivityScore"] = data_baru["FAF"] - data_baru["TUE"]
        data_baru["EatingScore"]   = data_baru["FCVC"] + data_baru["NCP"]

        bmi_value = data_baru["BMI"].values[0]
        data_baru = data_baru.drop(columns=["BMI"])

        prediksi   = rf_model.predict(data_baru)[0]
        proba_all  = rf_model.predict_proba(data_baru)[0]
        presentase = max(proba_all)

        warna = TEAL  
        if "Obesity" in prediksi:
            warna = "#e53935"
        elif "Overweight" in prediksi:
            warna = "#fb8c00"

        label_map = {
            "Normal_Weight":      "Normal Weight",
            "Insufficient_Weight":"Insufficient Weight",
            "Overweight_Level_I": "Overweight Level I",
            "Overweight_Level_II":"Overweight Level II",
            "Obesity_Type_I":     "Obesity Type I",
            "Obesity_Type_II":    "Obesity Type II",
            "Obesity_Type_III":   "Obesity Type III",
        }
        prediksi_tampil = label_map.get(prediksi, prediksi)

        # Icon untuk hero result
        icon_map = {
            "Insufficient_Weight": "🥗",
            "Normal_Weight":       "💚",
            "Overweight_Level_I":  "⚠️",
            "Overweight_Level_II": "⚠️",
            "Obesity_Type_I":      "🚨",
            "Obesity_Type_II":     "🚨",
            "Obesity_Type_III":    "🆘",
        }
        icon_pred = icon_map.get(prediksi, "🩺")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="dot"></span>Hasil Prediksi</div>', unsafe_allow_html=True)

        # ---------- HERO RESULT ----------
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {warna} 0%, {NAVY_DARK} 130%);
            border-radius: 22px; padding: 34px 38px; color: white;
            box-shadow: 0 20px 50px rgba(0,0,0,0.18);
            position: relative; overflow: hidden;">
            <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">
                <div style="font-size:64px;line-height:1">{icon_pred}</div>
                <div style="flex:1;min-width:240px">
                    <div style="opacity:0.85;font-size:12px;letter-spacing:2px;text-transform:uppercase">Hasil Prediksi</div>
                    <h2 style="margin:6px 0;font-size:34px;font-weight:800">{prediksi_tampil}</h2>
                    <div style="opacity:0.92;font-size:14px">Confidence model: <b>{presentase*100:.2f}%</b></div>
                </div>
                <div style="text-align:center;background:rgba(255,255,255,0.12);
                            padding:16px 24px;border-radius:16px;backdrop-filter:blur(8px)">
                    <div style="font-size:12px;opacity:0.85;letter-spacing:1px">BMI ANDA</div>
                    <div style="font-size:42px;font-weight:800;line-height:1">{bmi_value:.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------- ST.METRICS ----------
        avg_per_kategori = {
            "Insufficient_Weight": {"weight": 47.0, "faf": 1.5, "ch2o": 1.8},
            "Normal_Weight":       {"weight": 67.0, "faf": 2.0, "ch2o": 2.2},
            "Overweight_Level_I":  {"weight": 78.0, "faf": 1.5, "ch2o": 2.0},
            "Overweight_Level_II": {"weight": 86.0, "faf": 1.2, "ch2o": 1.9},
            "Obesity_Type_I":      {"weight": 97.0, "faf": 1.0, "ch2o": 1.8},
            "Obesity_Type_II":     {"weight": 111.0,"faf": 0.8, "ch2o": 1.7},
            "Obesity_Type_III":    {"weight": 128.0,"faf": 0.5, "ch2o": 1.6},
        }
        avg = avg_per_kategori.get(prediksi, {"weight": 67.0, "faf": 2.0, "ch2o": 2.0})

        delta_weight = weight - avg["weight"]
        delta_faf    = faf - avg["faf"]
        delta_ch2o   = ch2o - avg["ch2o"]

        st.markdown('<div class="section-title"><span class="dot"></span>Ringkasan Data Anda</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            label="BMI Score",
            value=f"{bmi_value:.1f}",
            delta=f"{bmi_value - 24.9:.1f} dari batas normal" if bmi_value > 24.9 else f"{bmi_value - 18.5:.1f} dari batas bawah",
            delta_color="inverse"
        )
        m2.metric(
            label="Berat Badan",
            value=f"{weight:.1f} kg",
            delta=f"{delta_weight:+.1f} kg dari rata-rata {prediksi_tampil}",
            delta_color="inverse"
        )
        m3.metric(
            label="Frekuensi Olahraga",
            value=f"{faf:.1f}x/minggu",
            delta=f"{delta_faf:+.1f} dari rata-rata",
            delta_color="normal"
        )
        m4.metric(
            label="Konsumsi Air",
            value=f"{ch2o:.1f} liter",
            delta=f"{delta_ch2o:+.1f} dari rata-rata",
            delta_color="normal"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------- GAUGE + DONUT ----------
        col_gauge, col_donut = st.columns(2)

        with col_gauge:
            with st.container(border=True):
                st.markdown("##### 📍 Posisi BMI Anda")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(bmi_value, 1),
                    number={"suffix": "", "font": {"size": 28, "color": NAVY}},
                    gauge={
                        "axis": {"range": [10, 50], "tickwidth": 1, "tickcolor": NAVY},
                        "bar":  {"color": warna, "thickness": 0.28},
                        "steps": [
                            {"range": [10, 18.5],  "color": "#B5D4F4"},
                            {"range": [18.5, 24.9],"color": "#A6E3C7"},
                            {"range": [24.9, 29.9],"color": "#FAC775"},
                            {"range": [29.9, 34.9],"color": "#F09595"},
                            {"range": [34.9, 39.9],"color": "#E24B4A"},
                            {"range": [39.9, 50],  "color": "#A32D2D"},
                        ],
                        "threshold": {
                            "line": {"color": warna, "width": 4},
                            "thickness": 0.75,
                            "value": bmi_value
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=260, margin=dict(t=20, b=10, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(
                        x=0.5, y=0.15, showarrow=False,
                        text="Underweight | Normal | Overweight | Obese",
                        font=dict(size=9, color="gray")
                    )]
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        with col_donut:
            with st.container(border=True):
                st.markdown("##### 🎯 Confidence per Kelas")
                kelas_labels = [label_map.get(c, c) for c in rf_model.classes_]
                fig_donut = go.Figure(go.Pie(
                    labels=kelas_labels,
                    values=[round(p * 100, 2) for p in proba_all],
                    hole=0.62,
                    textinfo="none",
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                    marker=dict(colors=[
                        TEAL, "#B5D4F4", "#FAC775", "#F09595",
                        "#E24B4A", "#A32D2D", "#791F1F"
                    ])
                ))
                fig_donut.update_layout(
                    height=260, margin=dict(t=20, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(font=dict(size=10), orientation="v"),
                    annotations=[dict(
                        text=f"<b>{presentase*100:.1f}%</b>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=20, color=warna)
                    )]
                )
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------- REKOMENDASI ----------
        st.markdown('<div class="section-title"><span class="dot"></span>🧠 Rekomendasi Kesehatan</div>', unsafe_allow_html=True)
        if prediksi == "Insufficient_Weight":
            st.info("⚠️ Berat badan Anda kurang. Tingkatkan asupan nutrisi dan konsultasikan ke dokter.")
        elif prediksi == "Normal_Weight":
            st.success("👍 Berat badan Anda normal. Pertahankan pola hidup sehat!")
        elif "Overweight" in prediksi:
            st.warning("⚠️ Anda mulai kelebihan berat badan. Atur pola makan & perbanyak olahraga.")
        elif "Obesity" in prediksi:
            st.error("🚨 Perlu perhatian serius. Segera konsultasikan ke tenaga medis.")
        else:
            st.info("Hasil belum dapat diklasifikasikan dengan jelas.")

        # ---------- SARAN AKTIVITAS ----------
        st.markdown('<div class="section-title"><span class="dot"></span>🏃 Saran Aktivitas</div>', unsafe_allow_html=True)
        aktivitas = {
            "Insufficient_Weight": ["🥗 Konsumsi makanan bergizi tinggi seperti protein, karbohidrat kompleks, dan lemak sehat","🏋️ Lakukan latihan beban ringan untuk membangun massa otot","🥛 Tambah asupan susu, telur, dan kacang-kacangan setiap hari","🛌 Pastikan tidur 7–9 jam per malam untuk mendukung pertumbuhan","🩺 Konsultasikan program makan dengan ahli gizi"],
            "Normal_Weight":       ["🚶 Pertahankan aktivitas jalan kaki minimal 30 menit setiap hari","🥦 Konsumsi sayur dan buah setidaknya 5 porsi per hari","💧 Minum air putih minimal 2 liter per hari","🏊 Lakukan olahraga aerobik seperti renang atau bersepeda 3x seminggu","😴 Jaga kualitas tidur dan kelola stres dengan baik"],
            "Overweight_Level_I":  ["🏃 Mulai jogging ringan 20–30 menit, 3–4 kali seminggu","🥗 Kurangi makanan tinggi gula dan lemak jenuh","🚴 Ganti transportasi dengan bersepeda atau jalan kaki jika memungkinkan","📱 Batasi penggunaan gadget dan gantikan dengan aktivitas fisik ringan","💧 Tingkatkan konsumsi air putih dan kurangi minuman manis"],
            "Overweight_Level_II": ["🏋️ Latihan kardio intensitas sedang minimal 45 menit, 4x seminggu","🍽️ Atur porsi makan dan hindari makan larut malam","🚶 Berjalan kaki setidaknya 8.000–10.000 langkah per hari","🧘 Tambahkan yoga atau stretching untuk mengurangi stres","🩺 Pertimbangkan konsultasi dengan dokter atau ahli gizi"],
            "Obesity_Type_I":      ["🩺 Segera konsultasikan dengan dokter untuk program penurunan berat badan","🏊 Olahraga ringan di air (aqua aerobik) untuk mengurangi beban sendi","🍱 Terapkan diet seimbang dengan porsi kecil tapi sering","📊 Pantau berat badan dan tekanan darah secara rutin","🚶 Mulai dengan berjalan kaki santai 15–20 menit per hari"],
            "Obesity_Type_II":     ["🏥 Wajib berkonsultasi dengan tenaga medis sebelum memulai program olahraga","🫀 Pantau kondisi jantung dan gula darah secara berkala","🥣 Ikuti program diet ketat yang dirancang oleh ahli gizi","🧘 Lakukan aktivitas fisik sangat ringan seperti stretching atau chair exercise","💊 Diskusikan kemungkinan terapi medis atau intervensi klinis dengan dokter"],
            "Obesity_Type_III":    ["🏥 Intervensi medis sangat dianjurkan, segera temui dokter spesialis","🛋️ Mulai dengan gerakan sederhana dari posisi duduk atau berbaring","🥗 Ikuti program nutrisi ketat yang dipantau tenaga medis","🫂 Cari dukungan psikologis untuk motivasi dan konsistensi","📋 Pertimbangkan program rehabilitasi medis yang terstruktur"],
        }
        saran_list = aktivitas.get(prediksi, [])
        cols_saran = st.columns(2)
        for i, saran in enumerate(saran_list):
            with cols_saran[i % 2]:
                st.markdown(f'<div class="tip-card">{saran}</div>', unsafe_allow_html=True)
# TAB 2 — INFORMASI
with tab2:
    # ---------- HERO ----------
    st.markdown(f"""
    <div class="hero">
        <span class="badge">● Health Insights</span>
        <h1>Informasi & Analisis Obesitas</h1>
        <p>Pelajari pengertian obesitas, eksplorasi dataset, dan evaluasi performa model
        machine learning yang digunakan dalam aplikasi HealthPredict.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- PENGERTIAN ----------
    st.markdown("""
    <div class="card-large">
        <h3>Pengertian Obesitas</h3>
        <p>Obesitas adalah kondisi medis yang ditandai dengan penumpukan lemak berlebih di dalam tubuh,
        yang dapat meningkatkan risiko berbagai penyakit seperti diabetes, hipertensi, dan penyakit jantung.</p>
        <p>Kondisi ini terjadi akibat ketidakseimbangan antara asupan energi dan aktivitas fisik, serta dipengaruhi
        oleh faktor genetik dan gaya hidup sehari-hari.</p>
        <p>Oleh karena itu, diperlukan suatu sistem yang dapat membantu mengidentifikasi tingkat obesitas secara cepat
        berdasarkan data yang dimasukkan oleh pengguna.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- STATISTIK ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>2.111</h3><p>Jumlah Data</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>16</h3><p>Jumlah Fitur</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>98%</h3><p>Akurasi Model</p></div>', unsafe_allow_html=True)

    # ---------- TENTANG SISTEM ----------
    st.markdown("""
    <div class="card-large">
        <h3>Tentang Sistem</h3>
        <p>Sistem ini menggunakan metode <b>machine learning Random Forest</b> untuk mengklasifikasikan tingkat obesitas
        berdasarkan data pengguna seperti usia, tinggi badan, berat badan, kebiasaan makan, dan aktivitas fisik.</p>
        <p>Dengan adanya sistem ini, pengguna dapat mengetahui kategori tingkat obesitas secara cepat dan cukup akurat,
        sehingga dapat menjadi dasar dalam menjaga dan meningkatkan kualitas kesehatan.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- PENJELASAN ISTILAH ----------
    st.markdown("""
    <div class="card-section">
        <h3>📖 Penjelasan Istilah</h3>
        <table class="tabel-istilah">
        <tr><td>Gender</td><td>: Jenis kelamin pengguna (Male / Female).</td></tr>
        <tr><td>Age</td><td>: Usia pengguna (dalam tahun).</td></tr>
        <tr><td>Height</td><td>: Tinggi badan dalam meter (contoh: 1.72, bukan 172).</td></tr>
        <tr><td>Weight</td><td>: Berat badan dalam kilogram.</td></tr>
        <tr><td>family_history_with_overweight</td><td>: Riwayat keluarga yang memiliki kelebihan berat badan (yes / no).</td></tr>
        <tr><td>FAVC</td><td>: Kebiasaan mengonsumsi makanan tinggi kalori (yes / no).</td></tr>
        <tr><td>FCVC</td><td>: Frekuensi konsumsi sayur (1 = jarang, 2 = kadang, 3 = sering).</td></tr>
        <tr><td>NCP</td><td>: Jumlah makan utama per hari (1–4 kali).</td></tr>
        <tr><td>CAEC</td><td>: Kebiasaan ngemil (no / Sometimes / Frequently / Always).</td></tr>
        <tr><td>CH2O</td><td>: Konsumsi air putih per hari (1 = &lt;1 liter, 2 = 1–2 liter, 3 = &gt;2 liter).</td></tr>
        <tr><td>FAF</td><td>: Frekuensi aktivitas fisik per minggu (0–3).</td></tr>
        <tr><td>TUE</td><td>: Durasi penggunaan gadget per hari (0–2).</td></tr>
        <tr><td>CALC</td><td>: Frekuensi konsumsi alkohol (no / Sometimes / Frequently / Always).</td></tr>
        <tr><td>MTRANS</td><td>: Jenis transportasi yang sering digunakan.</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    
#TAB 3 ANALISIS DATA
with tab3:
    # ANALISIS DATA 
    st.markdown('<div class="section-title"><span class="dot"></span>📈 Analisis Data</div>', unsafe_allow_html=True)
    st.write("Eksplorasi dan analisis dataset obesitas secara menyeluruh.")

    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    # ---------- VISUALISASI RINGKAS PLOTLY ----------
    st.markdown("### Visualisasi Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card-chart">', unsafe_allow_html=True)
        kategori = df["NObeyesdad"].value_counts().reset_index()
        kategori.columns = ["Kategori", "Jumlah"]
        fig1 = px.bar(kategori, x="Kategori", y="Jumlah",
                      title="<b>Kategori Obesitas</b>",
                      text="Jumlah", color_discrete_sequence=[NAVY])
        fig1.update_traces(textposition="outside", marker_line_width=0)
        fig1.update_xaxes(tickangle=-30)
        fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card-chart">', unsafe_allow_html=True)
        fig2 = px.pie(df, names="CALC", title="<b>Konsumsi Alkohol</b>", hole=0.5,
                      color_discrete_sequence=[NAVY, TEAL, "#5DCAA5", "#B5D4F4"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10, l=0, r=0))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig3 = px.scatter(df, x="Weight", y="Height", color="NObeyesdad",
        title="<b>Hubungan Berat dan Tinggi Badan per Kategori Obesitas</b>",
        labels={"Weight": "Berat Badan (kg)", "Height": "Tinggi Badan (m)", "NObeyesdad": "Kategori"},
        opacity=0.75, color_discrete_sequence=px.colors.qualitative.Bold)
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=60, b=20, l=0, r=0))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 1. LOAD DATASET ----------
    st.markdown("### 1. Load Dataset")
    with st.expander("Lihat Kode", expanded=False):
        st.code('df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")\ndf', language="python")
    st.dataframe(df, use_container_width=True)

    # ---------- 2. INFO DATASET ----------
    st.markdown("### 2. Informasi Dataset")
    with st.expander("Lihat Kode", expanded=False):
        st.code("df.info()\ndf.describe()", language="python")
    info_df = pd.DataFrame({"Kolom": df.columns, "Non-Null": df.notnull().sum().values, "Dtype": df.dtypes.values})
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    st.markdown("**Statistik Deskriptif:**")
    st.dataframe(df.describe(), use_container_width=True)

    # ---------- 3. HEAD & TAIL ----------
    st.markdown("### 3. Head & Tail Data")
    with st.expander("Lihat Kode", expanded=False):
        st.code("df.head()\ndf.tail()", language="python")
    tab_head, tab_tail = st.tabs(["Head (5 Baris Pertama)", "Tail (5 Baris Terakhir)"])
    with tab_head:
        st.dataframe(df.head(), use_container_width=True)
    with tab_tail:
        st.dataframe(df.tail(), use_container_width=True)

    # ---------- 4. MISSING VALUES ----------
    st.markdown("### 4. Pengecekan Missing Values")
    with st.expander("Lihat Kode", expanded=False):
        st.code("df.isna().sum()", language="python")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Kolom", "Jumlah Missing"]
    st.dataframe(missing, use_container_width=True, hide_index=True)

    # ---------- 5. VALUE COUNTS ----------
    st.markdown("### 5. Value Counts Kolom Kategorik")
    cat_cols = {
        "Gender": 'df["Gender"].value_counts()',
        "family_history_with_overweight": 'df["family_history_with_overweight"].value_counts()',
        "FAVC": 'df["FAVC"].value_counts()',
        "CAEC": 'df["CAEC"].value_counts()',
        "SMOKE": 'df["SMOKE"].value_counts()',
        "SCC": 'df["SCC"].value_counts()',
        "CALC": 'df["CALC"].value_counts()',
        "MTRANS": 'df["MTRANS"].value_counts()',
        "NObeyesdad": 'df["NObeyesdad"].value_counts()',
    }
    tabs_vc = st.tabs(list(cat_cols.keys()))
    for tab, (col_name, code_str) in zip(tabs_vc, cat_cols.items()):
        with tab:
            with st.expander("Lihat Kode", expanded=False):
                st.code(code_str, language="python")
            vc = df[col_name].value_counts().reset_index()
            vc.columns = [col_name, "Jumlah"]
            st.dataframe(vc, use_container_width=True, hide_index=True)

    # ---------- 6. VISUALISASI EDA ----------
    st.markdown("### 6. Visualisasi EDA")

    st.markdown("#### a. Distribusi Tingkat Obesitas")
    with st.expander("Lihat Kode", expanded=False):
        st.code('sns.countplot(data=df, x="NObeyesdad")', language="python")
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="NObeyesdad", ax=ax_a, color=NAVY)
    ax_a.set_title("Distribusi Tingkat Obesitas")
    ax_a.set_xlabel("Kategori Obesitas"); ax_a.set_ylabel("Jumlah")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    st.pyplot(fig_a); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### b. Pengaruh Riwayat Keluarga terhadap Obesitas")
    with st.expander("Lihat Kode", expanded=False):
        st.code('sns.countplot(data=df, x="family_history_with_overweight", hue="NObeyesdad", palette="Set1")', language="python")
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="family_history_with_overweight", hue="NObeyesdad", palette="Set1", ax=ax_b)
    ax_b.set_title("Pengaruh Riwayat Keluarga terhadap Obesitas")
    ax_b.set_xlabel("Riwayat Keluarga"); ax_b.set_ylabel("Jumlah")
    ax_b.legend(title="Kategori", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); st.pyplot(fig_b); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### c. Konsumsi Makanan Tinggi Kalori vs Obesitas")
    with st.expander("Lihat Kode", expanded=False):
        st.code('sns.countplot(data=df, x="FAVC", hue="NObeyesdad", palette="coolwarm")', language="python")
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="FAVC", hue="NObeyesdad", palette="coolwarm", ax=ax_c)
    ax_c.set_title("Konsumsi Makanan Tinggi Kalori vs Obesitas")
    ax_c.set_xlabel("FAVC"); ax_c.set_ylabel("Jumlah")
    ax_c.legend(title="Kategori", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); st.pyplot(fig_c); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### d. Distribusi BMI terhadap Tingkat Obesitas")
    with st.expander("Lihat Kode", expanded=False):
        st.code('df["BMI"] = df["Weight"] / (df["Height"]**2)\nsns.boxplot(data=df, x="NObeyesdad", y="BMI", showfliers=False)', language="python")
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_d, ax_d = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="NObeyesdad", y="BMI", showfliers=False, ax=ax_d, color=TEAL)
    ax_d.set_title("Distribusi BMI terhadap Tingkat Obesitas")
    ax_d.set_xlabel("Kategori Obesitas"); ax_d.set_ylabel("BMI")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    st.pyplot(fig_d); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### e. Konsumsi Air terhadap Tingkat Obesitas")
    with st.expander("Lihat Kode", expanded=False):
        st.code('sns.boxplot(data=df, x="NObeyesdad", y="CH2O", showfliers=False)', language="python")
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_e, ax_e = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="NObeyesdad", y="CH2O", showfliers=False, ax=ax_e, color=NAVY)
    ax_e.set_title("Konsumsi Air terhadap Tingkat Obesitas")
    ax_e.set_xlabel("Kategori Obesitas"); ax_e.set_ylabel("CH2O")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    st.pyplot(fig_e); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### f. Jumlah Makan Utama vs Obesitas")
    with st.expander("Lihat Kode", expanded=False):
        st.code('sns.boxplot(data=df, x="NObeyesdad", y="NCP", showfliers=False)', language="python")
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_f, ax_f = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="NObeyesdad", y="NCP", showfliers=False, ax=ax_f, color=TEAL)
    ax_f.set_title("Jumlah Makan Utama vs Obesitas")
    ax_f.set_xlabel("Kategori Obesitas"); ax_f.set_ylabel("NCP")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    st.pyplot(fig_f); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### g. Heatmap Korelasi Antar Fitur Numerik")
    with st.expander("Lihat Kode", expanded=False):
        st.code('corr = df[numeric_cols].corr()\nsns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")', language="python")
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    numeric_cols = ["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    corr = df[numeric_cols].corr()
    fig_g, ax_g = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True, ax=ax_g)
    ax_g.set_title("Heatmap Korelasi Antar Fitur Numerik", fontsize=14, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig_g); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 7. FEATURE ENGINEERING ----------
    st.markdown("### 7. Feature Engineering")
    with st.expander("Lihat Kode", expanded=False):
        st.code('df["BMI_Category"] = df["BMI"].apply(bmi_category)\ndf["ActivityScore"] = df["FAF"] - df["TUE"]\ndf["EatingScore"] = df["FCVC"] + df["NCP"]', language="python")
    df["BMI_Category"]  = df["BMI"].apply(bmi_category)
    df["ActivityScore"] = df["FAF"] - df["TUE"]
    df["EatingScore"]   = df["FCVC"] + df["NCP"]
    st.dataframe(df[["BMI", "BMI_Category", "ActivityScore", "EatingScore"]].head(10), use_container_width=True)

    # ---------- 8. PERBANDINGAN MODEL ----------
    st.markdown("### 8. Perbandingan Akurasi 3 Model")
    with st.expander("Lihat Kode", expanded=False):
        st.code("""model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies  = [acc_lr, acc_dt, acc_rf]
plt.bar(model_names, accuracies)""", language="python")
    model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
    accuracies  = [0.823, 0.961, 0.981]
    st.markdown('<div class="card-chart">', unsafe_allow_html=True)
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    bars = ax_acc.bar(model_names, accuracies, color=[NAVY, "#5b6fa8", TEAL])
    ax_acc.set_title("Perbandingan Akurasi 3 Model", fontsize=14, fontweight="bold")
    ax_acc.set_ylabel("Akurasi"); ax_acc.set_ylim(0.7, 1.05)
    ax_acc.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout(); st.pyplot(fig_acc); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 9. FEATURE IMPORTANCE ----------
    st.markdown("### 9. Feature Importance (Random Forest)")
    with st.expander("Lihat Kode", expanded=False):
        st.code('importances = rf_model.named_steps["model"].feature_importances_', language="python")
    try:
        importances = rf_model.named_steps["model"].feature_importances_
        try:
            feature_names = rf_model.named_steps["preprocessing"].get_feature_names_out()
        except Exception:
            try:
                feature_names = rf_model.feature_names_in_
            except Exception:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
        feature_names = list(feature_names)
        indices = np.argsort(importances)[::-1][:15]
        st.markdown('<div class="card-chart">', unsafe_allow_html=True)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        ax_fi.barh([feature_names[i] for i in indices][::-1], importances[indices][::-1], color=TEAL)
        ax_fi.set_title("Top 15 Fitur Paling Berpengaruh terhadap Prediksi Tingkat Obesitas", fontsize=13, fontweight="bold")
        ax_fi.set_xlabel("Importance Score")
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Feature importance tidak dapat ditampilkan: {e}")

    # EVALUASI MODEL (logika asli)
    st.markdown("---")
    st.markdown('<div class="section-title"><span class="dot"></span>🧪 Evaluasi Performa Model Machine Learning</div>', unsafe_allow_html=True)

    df_eval = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    df_eval["BMI"]           = df_eval["Weight"] / (df_eval["Height"] ** 2)
    df_eval["BMI_Category"]  = df_eval["BMI"].apply(bmi_category)
    df_eval["ActivityScore"] = df_eval["FAF"] - df_eval["TUE"]
    df_eval["EatingScore"]   = df_eval["FCVC"] + df_eval["NCP"]
    df_eval["CAEC"]          = df_eval["CAEC"].str.strip()
    df_eval["CALC"]          = df_eval["CALC"].str.strip()

    X = df_eval[["Gender", "Age", "Height", "Weight", "BMI_Category", "family_history_with_overweight",
                 "FAVC", "FCVC", "NCP", "CAEC", "CH2O", "FAF", "TUE", "CALC", "MTRANS", "EatingScore", "ActivityScore"]]
    y = df_eval["NObeyesdad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred      = rf_model.predict(X_test)
    acc         = accuracy_score(y_test, y_pred)
    report      = classification_report(y_test, y_pred, output_dict=True)
    macro_f1    = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card-metric"><p>Accuracy Score</p><h2>{acc:.2%}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card-metric"><p>Macro Avg F1</p><h2>{macro_f1:.2%}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card-metric"><p>Weighted Avg F1</p><h2>{weighted_f1:.2%}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Classification Report")
    report_df  = pd.DataFrame(report).transpose()
    classes    = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
                  "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    rows_kelas = [r for r in classes if r in report_df.index]
    kelas_df   = report_df.loc[rows_kelas, ["precision", "recall", "f1-score", "support"]].copy()
    kelas_df.index.name = "Kelas"
    kelas_df.columns    = ["Precision", "Recall", "F1-Score", "Support"]
    kelas_df["Support"] = kelas_df["Support"].astype(int)
    kelas_df[["Precision", "Recall", "F1-Score"]] = kelas_df[["Precision", "Recall", "F1-Score"]].map(lambda x: f"{x:.2f}")
    st.dataframe(kelas_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    fig_cm, ax_cm = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=rf_model.classes_, yticklabels=rf_model.classes_,
                ax=ax_cm, linewidths=0.5, linecolor="white")
    ax_cm.set_xlabel("Prediksi", fontsize=12)
    ax_cm.set_ylabel("Aktual", fontsize=12)
    ax_cm.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(); st.pyplot(fig_cm); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Mean Score")
    mean_precision = report_df.loc[rows_kelas, "precision"].mean()
    mean_recall    = report_df.loc[rows_kelas, "recall"].mean()
    mean_f1        = report_df.loc[rows_kelas, "f1-score"].mean()
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Mean Precision", f"{mean_precision:.2%}")
    with mc2:
        st.metric("Mean Recall", f"{mean_recall:.2%}")
    with mc3:
        st.metric("Mean F1-Score", f"{mean_f1:.2%}")

# TAB 4 — ABOUT ME
with tab4:
    with tab4:
        st.markdown(f"""
        <div class="hero">
            <h1>Tentang Saya</h1>
           <p> Halo! Saya Silfi Dwi Ramadani, siswa kelas 11 jurusan Rekayasa Perangkat Lunak.
               HealthPredict dikembangkan sebagai project persiapan PKL sekaligus bentuk
               penerapan kemampuan dalam membangun aplikasi interaktif berbasis web.
          </p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.image("img.jpg")
        st.markdown("""
            <h2 style="margin:8px 0 0 0;color:#2c3e7a">Silfi Dwi Ramadani</h2>
            <div style="margin-top:18px">
                <a href="mailto:silfiramadani86@gmail.com" class="social-btn">🌐 Email</a>
                <a href="https://instagram.com/silfirmdni_" class="social-btn">📸 Instagram</a>
                <a href="https://github.com/silfirmdni14" class="social-btn">🐙 GitHub</a>
            </div>
        </div>
        """, unsafe_allow_html=True)    
