import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Sayfa Yapılandırması
st.set_page_config(page_title="Sigara Kullanımı Tahmin Sistemi", layout="wide")
st.title("🩺 Biyometrik Veriler ile Sigara Kullanımı Tahmini")

# 2. Modeli Yükle
@st.cache_resource
def load_model():
    return joblib.load('lgbm_smoking_model.pkl')

model = load_model()

# Modelin eğitimde gördüğü sütunların listesini alalım (smoking hariç 31 tane)
# Bu liste sıralamayı garantiye alacaktır.
model_features = [
    'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
    'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic',
    'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
    'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST',
    'ALT', 'Gtp', 'dental caries', 'log_Gtp', 'log_triglyceride', 
    'log_ALT', 'log_LDL', 'bmi', 'ast_alt_ratio', 'chol_hdl_ratio',
    'is_high_hemo', 'is_high_gtp'
]

# 3. Kullanıcı Giriş Paneli
st.sidebar.header("Kullanıcı Verilerini Girin")

def user_input_features():
    # Sayısal girişler
    age = st.sidebar.slider("Yaş", 20, 85, 40)
    height = st.sidebar.slider("Boy (cm)", 135, 190, 170)
    weight = st.sidebar.slider("Kilo (kg)", 30, 130, 70)
    waist = st.sidebar.slider("Bel Çevresi (cm)", 50, 130, 85)
    hemoglobin = st.sidebar.number_input("Hemoglobin", 4.0, 21.0, 15.0)
    gtp = st.sidebar.number_input("Gtp", 1, 1000, 35)
    triglyceride = st.sidebar.number_input("Triglyceride", 1, 800, 120)
    alt = st.sidebar.number_input("ALT", 1, 2900, 25)
    ast = st.sidebar.number_input("AST", 1, 800, 25)
    ldl = st.sidebar.number_input("LDL", 1, 1800, 115)
    hdl = st.sidebar.number_input("HDL", 1, 150, 55)
    cholesterol = st.sidebar.number_input("Toplam Kolesterol", 70, 400, 200)
    dental_caries = st.sidebar.selectbox("Diş Çürüğü Var mı?", [0, 1])
    
    # Modelin beklediği ham veriler
    data = {
        'age': age, 'height(cm)': height, 'weight(kg)': weight, 'waist(cm)': waist,
        'eyesight(left)': 1.0, 'eyesight(right)': 1.0, 'hearing(left)': 1.0, 
        'hearing(right)': 1.0, 'systolic': 120, 'relaxation': 80, 
        'fasting blood sugar': 100, 'Cholesterol': cholesterol, 'triglyceride': triglyceride,
        'HDL': hdl, 'LDL': ldl, 'hemoglobin': hemoglobin, 'Urine protein': 1.0, 
        'serum creatinine': 0.9, 'AST': ast, 'ALT': alt, 'Gtp': gtp, 
        'dental caries': dental_caries
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Feature Engineering (SADECE train_df sütunlarında olanlar!)
def preprocess(df):
    df = df.copy()
    # Log Dönüşümleri
    df['log_Gtp'] = np.log1p(df['Gtp'])
    df['log_triglyceride'] = np.log1p(df['triglyceride'])
    df['log_ALT'] = np.log1p(df['ALT'])
    df['log_LDL'] = np.log1p(df['LDL'])
    
    # Oranlar
    df['bmi'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    df['ast_alt_ratio'] = df['AST'] / df['ALT']
    df['chol_hdl_ratio'] = df['Cholesterol'] / df['HDL']
    
    # Risk Bayrakları
    df['is_high_hemo'] = (df['hemoglobin'] > 16).astype(int)
    df['is_high_gtp'] = (df['Gtp'] > 60).astype(int)
    
    # NOT: 'hemo_gtp_risk' sütununu sildim çünkü senin train_df sütun listende yok!
    
    # ÖNEMLİ: Sütunları modelin beklediği SIRAYA sokuyoruz
    df = df[model_features]
    return df

processed_df = preprocess(input_df)

# 5. Tahmin Arayüzü
st.subheader("📊 Analiz Sonucu")

if st.button("Tahmini Hesapla"):
    # Tahmin
    prediction_proba = model.predict_proba(processed_df)[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Sigara İçme Olasılığı", value=f"%{prediction_proba*100:.2f}")
    with col2:
        if prediction_proba > 0.5:
            st.error("Tahmin: SİGARA İÇİYOR")
        else:
            st.success("Tahmin: SİGARA İÇMİYOR")
    st.progress(prediction_proba)
else:
    st.info("Değerleri girip butona basın.")