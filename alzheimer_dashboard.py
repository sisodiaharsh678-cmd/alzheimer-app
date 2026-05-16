import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
 
# ============================================================
# RUN YOUR PROJECT FILE — loads all trained models & scaler
# ============================================================
exec(open("final_year.py").read())
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Alzheimer's Early Detection",
    page_icon="🧠",
    layout="centered"
)
 
# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main { max-width: 800px; }
    .title-box {
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .title-box h1 { font-size: 2rem; margin: 0; }
    .title-box p  { font-size: 1rem; margin: 8px 0 0; opacity: 0.85; }
    .section-header {
        background: #F3F0FF;
        border-left: 5px solid #7C3AED;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 16px;
        color: #4C1D95;
        margin: 24px 0 14px;
    }
    .result-positive {
        background: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .result-negative {
        background: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #9CA3AF;
        font-size: 12px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="title-box">
    <h1>🧠 Alzheimer's Early Detection</h1>
    <p>Enter patient details below to check the risk of Alzheimer's disease</p>
    <p style="font-size:12px; opacity:0.7;">Final Year Project · BSc (Hons) Computing Science · PSB Academy / Coventry University</p>
</div>
""", unsafe_allow_html=True)
 
st.info(f"🤖 Powered by **{best_model}** — Best performing model (highest ROC-AUC)")
 
# ============================================================
# PATIENT DETAILS FORM
# ============================================================
st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age       = st.number_input("Age", min_value=60, max_value=90, value=72)
with col2:
    gender    = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
with col3:
    ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3],
                             format_func=lambda x: ["Caucasian","African American","Asian","Other"][x])
 
col1, col2 = st.columns(2)
with col1:
    education = st.selectbox("Education Level", [0, 1, 2, 3],
                             format_func=lambda x: ["None","High School","Bachelor's","Higher"][x])
with col2:
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)
 
# ============================================================
st.markdown('<div class="section-header">🏥 Lifestyle Factors</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    smoking  = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col2:
    alcohol  = st.slider("Alcohol Consumption", 0.0, 20.0, 5.0)
with col3:
    physical = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 3.0)
 
col1, col2 = st.columns(2)
with col1:
    diet     = st.slider("Diet Quality Score (0–10)", 0.0, 10.0, 5.0)
with col2:
    sleep    = st.slider("Sleep Quality Score (4–10)", 4.0, 10.0, 7.0)
 
# ============================================================
st.markdown('<div class="section-header">❤️ Medical History</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    family_history  = st.selectbox("Family History of Alzheimer's", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col2:
    cardiovascular  = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col3:
    diabetes        = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
 
col1, col2, col3 = st.columns(3)
with col1:
    depression      = st.selectbox("Depression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col2:
    head_injury     = st.selectbox("Head Injury", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col3:
    hypertension    = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
 
# ============================================================
st.markdown('<div class="section-header">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    systolic_bp  = st.slider("Systolic BP (mmHg)", 90, 180, 120)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
    chol_total   = st.slider("Total Cholesterol (mg/dL)", 150.0, 300.0, 200.0)
with col2:
    chol_ldl     = st.slider("LDL Cholesterol", 50.0, 200.0, 100.0)
    chol_hdl     = st.slider("HDL Cholesterol", 20.0, 100.0, 50.0)
    chol_trig    = st.slider("Triglycerides", 50.0, 400.0, 150.0)
 
# ============================================================
st.markdown('<div class="section-header">🧪 Cognitive & Functional Assessment</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    mmse              = st.slider("MMSE Score (0–30)", 0.0, 30.0, 24.0,
                                  help="Mini-Mental State Examination. Lower = more cognitive impairment.")
    functional_assess = st.slider("Functional Assessment (0–10)", 0.0, 10.0, 5.0)
    memory_complaints = st.selectbox("Memory Complaints", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col2:
    behavioral        = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    adl               = st.slider("ADL Score (0–10)", 0.0, 10.0, 5.0,
                                  help="Activities of Daily Living. Lower = more difficulty.")
    confusion         = st.selectbox("Confusion", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
 
col1, col2 = st.columns(2)
with col1:
    disorientation    = st.selectbox("Disorientation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col2:
    personality       = st.selectbox("Personality Changes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
 
col1, col2 = st.columns(2)
with col1:
    difficulty_tasks  = st.selectbox("Difficulty with Tasks", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col2:
    forgetfulness     = st.selectbox("Forgetfulness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
 
# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("---")
predict_btn = st.button("🔮 Check Alzheimer's Risk", use_container_width=True, type="primary")
 
if predict_btn:
    # Build input with all features filled to median
    input_dict = {col: [float(df[col].median())] for col in X.columns}
 
    # Override with patient inputs
    user_vals = {
        'Age': age, 'Gender': gender, 'Ethnicity': ethnicity,
        'EducationLevel': education, 'BMI': bmi, 'Smoking': smoking,
        'AlcoholConsumption': alcohol, 'PhysicalActivity': physical,
        'DietQuality': diet, 'SleepQuality': sleep,
        'FamilyHistoryAlzheimers': family_history,
        'CardiovascularDisease': cardiovascular, 'Diabetes': diabetes,
        'Depression': depression, 'HeadInjury': head_injury,
        'Hypertension': hypertension, 'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp, 'CholesterolTotal': chol_total,
        'CholesterolLDL': chol_ldl, 'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_trig, 'MMSE': mmse,
        'FunctionalAssessment': functional_assess,
        'MemoryComplaints': memory_complaints,
        'BehavioralProblems': behavioral, 'ADL': adl,
        'Confusion': confusion, 'Disorientation': disorientation,
        'PersonalityChanges': personality,
        'DifficultyCompletingTasks': difficulty_tasks,
        'Forgetfulness': forgetfulness
    }
 
    for col, val in user_vals.items():
        if col in input_dict:
            input_dict[col] = [float(val)]
 
    input_df     = pd.DataFrame(input_dict)
    input_scaled = scaler.transform(input_df)
 
    prediction   = best_model.predict(input_scaled)[0]
    probability  = best_model.predict_proba(input_scaled)[0][1]
 
    st.markdown("---")
    st.subheader("📋 Result")
 
    # Result box
    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <h2>🔴 High Risk — Alzheimer's Detected</h2>
            <p style="font-size:18px;">This patient shows <b>{probability*100:.1f}%</b> probability of Alzheimer's disease.</p>
            <p>⚠️ Please consult a neurologist or specialist immediately for further evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <h2>🟢 Low Risk — No Alzheimer's Detected</h2>
            <p style="font-size:18px;">This patient shows <b>{probability*100:.1f}%</b> probability of Alzheimer's disease.</p>
            <p>✅ No immediate concern. Continue regular health checkups.</p>
        </div>
        """, unsafe_allow_html=True)
 
    # Gauge chart
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(probability * 100, 1),
        title = {'text': "Alzheimer's Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar':  {'color': '#7C3AED'},
            'steps': [
                {'range': [0,  30],  'color': '#D1FAE5'},
                {'range': [30, 60],  'color': '#FEF3C7'},
                {'range': [60, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line':      {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value':     50
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
 
    # Key factors
    st.subheader("🔍 Key Risk Factors Entered")
    risk_summary = pd.DataFrame({
        'Factor':       ['MMSE Score', 'Functional Assessment', 'ADL Score',
                         'Memory Complaints', 'Family History', 'Depression', 'Age'],
        'Value':        [mmse, functional_assess, adl,
                         'Yes' if memory_complaints else 'No',
                         'Yes' if family_history else 'No',
                         'Yes' if depression else 'No',
                         age],
        'Risk Level':   [
            '🔴 High' if mmse < 20 else ('🟡 Medium' if mmse < 25 else '🟢 Low'),
            '🔴 High' if functional_assess < 4 else ('🟡 Medium' if functional_assess < 7 else '🟢 Low'),
            '🔴 High' if adl < 4 else ('🟡 Medium' if adl < 7 else '🟢 Low'),
            '🔴 High' if memory_complaints else '🟢 Low',
            '🔴 High' if family_history else '🟢 Low',
            '🟡 Medium' if depression else '🟢 Low',
            '🔴 High' if age >= 80 else ('🟡 Medium' if age >= 70 else '🟢 Low')
        ]
    })
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)
 
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")
 
# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🧠 Alzheimer's Early Detection System · Final Year Project<br>
    BSc (Hons) Computing Science · PSB Academy / Coventry University<br>
    Built with Python · Scikit-learn · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)