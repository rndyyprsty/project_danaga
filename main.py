import streamlit as st
import pandas as pd
import os
import time
from model.model_trainer import ModelTrainer
from model.predictor import Predictor
import plotly.graph_objects as go

# konfigurasi page
st.set_page_config(
    page_title="Danaga",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body, .stApp, .css-1v0mbdj, .css-ffhzg2, .css-1d391kg, .css-qrbaxs, .stText, .stMarkdown, .stSidebar {
        color: red;
    }

    label {
        color: red;
        font-weight: bold;
    }

    h1, h2, h3 {
        color: red;
    }

</style>
""", unsafe_allow_html=True)

#sidebar
def display_sidebar():
    st.sidebar.title("Navigasi")
    st.sidebar.markdown("### Tentang Aplikasi")
    st.sidebar.write("Danaga adalah aplikasi prediksi waktu pelunasan hutang.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made in Heaven")

#header
def display_header():
    st.markdown("<h1 style='text-align: center;'>Danaga</h1>", unsafe_allow_html=True)
    st.divider()

#kolom inputan
def show_input_form():
    st.subheader("Masukkan keuanganmu saat ini")
    with st.form("input_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            total_assets = st.text_input("Total Aset (Rp)", placeholder="Contoh: 50000000")
            monthly_income = st.text_input("Penghasilan Bulanan (Rp)", placeholder="Contoh: 8000000")

        with col2:
            monthly_expenses = st.text_input("Pengeluaran Bulanan (Rp)", placeholder="Contoh: 4000000")
            total_debt = st.text_input("Total Hutang (Rp)", placeholder="Contoh: 20000000")

        submitted = st.form_submit_button("Analisis Finansial")
    
    return {
        "submitted": submitted,
        "total_assets": total_assets,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "total_debt": total_debt
    }

#cek input valid apa nggak
def validate_input(total_assets, monthly_income, monthly_expenses, total_debt):
    errors = []
    if not total_assets.isdigit():
        errors.append("Total Aset harus berupa angka.")
    if not monthly_income.isdigit():
        errors.append("Penghasilan Bulanan harus berupa angka.")
    if not monthly_expenses.isdigit():
        errors.append("Pengeluaran Bulanan harus berupa angka.")
    if not total_debt.isdigit():
        errors.append("Total Hutang harus berupa angka.")
    return errors

#tampil hasil
def show_results(total_assets, monthly_income, monthly_expenses, total_debt):
    # format input ke dataframe
    sample = pd.DataFrame([{
        'total_aset': float(total_assets.replace(',', '')),
        'penghasilan_bulanan': float(monthly_income.replace(',', '')),
        'pengeluaran_bulanan': float(monthly_expenses.replace(',', '')),
        'total_hutang': float(total_debt.replace(',', ''))
    }])

    #load model
    model_path = 'C:/Users/Randy/Documents/Kampus/SMST 4/PROJECT_DANAGA/artifacts/model.pkl'
    predictor = Predictor(model_path)
    predictor.load_model()
    result = predictor.predict(sample)

    #hasil predik
    st.success(f"Estimasi waktu pelunasan hutang kamu: **{result[0]:.2f} bulan**")

    #insight finansial
    remaining_income = float(monthly_income.replace(',', '')) - float(monthly_expenses.replace(',', ''))
    st.markdown("### Insight Finansial Kamu")
    st.markdown(f"""
    - **Sisa penghasilan per bulan:** Rp {remaining_income:,.2f}
    - **Total hutang:** Rp {float(total_debt):,.2f}
    - **Total aset:** Rp {float(total_assets):,.2f}
    """)

    #visualisasi
    fig = go.Figure(go.Pie(
        labels=["Pengeluaran", "Sisa Penghasilan"],
        values=[float(monthly_expenses.replace(',', '')), remaining_income],
        marker=dict(colors=["#FF6B6B", "#1DD1A1"]),
        hole=0.4
    ))
    fig.update_layout(title_text="Distribusi Penghasilan Bulanan", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def main():
    display_sidebar()
    display_header()

    st.markdown("<p class='red-text'>⚠️ Semua input harus berupa angka tanpa titik/koma.</p>", unsafe_allow_html=True)

    input_data = show_input_form()

    if input_data["submitted"]:
        errors = validate_input(
            input_data["total_assets"],
            input_data["monthly_income"],
            input_data["monthly_expenses"],
            input_data["total_debt"]
        )

        if errors:
            st.markdown("<div class='red-text'><strong>Terdapat kesalahan pada input:</strong></div>", unsafe_allow_html=True)
            for error in errors:
                st.markdown(f"<div class='red-text'>- {error}</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Sedang menganalisis data finansial Anda..."):
                time.sleep(1)
                show_results(
                    input_data["total_assets"],
                    input_data["monthly_income"],
                    input_data["monthly_expenses"],
                    input_data["total_debt"]
                )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center">
            <p>Project Danaga &copy; 2025</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()