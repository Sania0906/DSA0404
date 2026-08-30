import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MediCost - Premium Healthcare AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR PREMIUM LOOK
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #12355b, #4361ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
    }

    .card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0px 8px 32px rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }

    .login-box {
        max-width: 450px;
        margin: 100px auto;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0px 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
    }

    .result-box {
        background: linear-gradient(135deg, #eefaf2 0%, #dcf0e2 100%);
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        border: 2px solid #9ad6ae;
        box-shadow: 0px 10px 30px rgba(21, 115, 71, 0.15);
    }

    .result-money {
        font-size: 54px;
        font-weight: 900;
        color: #157347;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .treatment-badge {
        display: inline-block;
        padding: 8px 16px;
        background-color: #12355b;
        color: white;
        border-radius: 50px;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #667085;
        padding: 30px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_dataset():
    try:
        data = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Keep insurance.csv in the same folder as this Python file.")
        st.stop()
    data.columns = [str(col).strip().lower() for col in data.columns]
    required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    if not all(col in data.columns for col in required_columns):
        st.error(f"Missing required columns in dataset.")
        st.stop()
    data = data[required_columns].dropna()
    return data

df = load_dataset()

# ============================================================
# MACHINE LEARNING MODEL (Upgraded to RandomForest)
# ============================================================
@st.cache_resource
def train_model(data):
    X = data.drop(columns=["charges"])
    y = data["charges"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"]),
            ("num", "passthrough", ["age", "bmi", "children"])
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    return model, metrics

model, metrics = train_model(df)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def predict_charges(age, sex, bmi, children, smoker, region):
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, 
        "children": children, "smoker": smoker, "region": region
    }])
    return max(0, float(model.predict(patient)[0]))

def get_category_and_treatment(charges):
    if charges < 5000:
        return "Low 🟢", "🩺 Routine Checkup / Preventive Care"
    elif charges < 15000:
        return "Moderate 🟡", "💊 Specialized Therapy / Minor Procedure"
    elif charges < 30000:
        return "High 🟠", "🚑 Emergency Care / Moderate Surgery"
    else:
        return "Critical 🔴", "🏥 Complex Surgery / Critical Care"

# ============================================================
# LOGIN PAGE
# ============================================================
def login_page():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center;color:#12355b;font-weight:800;">🏥 MediCost AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#667085;margin-bottom:30px;">Next-Gen Healthcare Cost Prediction</p>', unsafe_allow_html=True)
    
    username = st.text_input("👤 Username")
    password = st.text_input("🔐 Password", type="password")
    
    if st.button("🔓 SECURE LOGIN", use_container_width=True, type="primary"):
        if not username.strip() or not password.strip():
            st.error("Please enter both Username and Password.")
        else:
            if "app_password" not in st.session_state:
                st.session_state.app_password = password
            
            if password == st.session_state.app_password:
                st.session_state.logged_in = True
                st.session_state.doctor_name = username.strip()
                st.rerun()
            else:
                st.error("Invalid Password.")
            
    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# 🏥 MediCost AI")
    st.caption("Premium Healthcare Analytics Portal")
    st.success(f"👨‍⚕️ Logged in as **{st.session_state.doctor_name}**")
    st.markdown("---")
    
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "👤 Patient Registration",
        "📋 Patient History",
        "📊 Advanced Analytics",
        "📈 AI Model Performance"
    ])
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">AI Doctor Dashboard</div>', unsafe_allow_html=True)
    
    st.write(f"Welcome back, **{st.session_state.doctor_name}**. Here's your clinic's overview.")
    
    total_patients = len(st.session_state.history)
    avg_charges = np.mean([r["Predicted Charges"] for r in st.session_state.history]) if total_patients > 0 else 0
    high_cost = sum(1 for r in st.session_state.history if "High" in r["Category"] or "Critical" in r["Category"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👤 Total Patients", total_patients)
    col2.metric("🔮 AI Predictions", total_patients)
    col3.metric("💰 Average Est. Cost", f"₹ {avg_charges:,.0f}")
    col4.metric("🔴 Intensive Care Cases", high_cost)
    
    st.markdown("### ⚡ Quick Access")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>👤 Register Patient</h4><p>Input vitals and get instant AI-driven cost & treatment predictions.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>📊 Analytics</h4><p>Dive deep into 3D visualizations and healthcare cost distributions.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>📈 Model Accuracy</h4><p>Review our RandomForest model boasting an R² score of {:.3f}.</p></div>'.format(metrics["r2"]), unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("### 🕒 Recent Patient Scans")
        recent = pd.DataFrame(st.session_state.history[-5:]).iloc[::-1]
        st.dataframe(recent[["Patient ID", "Patient Name", "Predicted Charges", "Treatment Type", "Date"]], use_container_width=True, hide_index=True)

    st.markdown("### 🔐 Account Settings")
    with st.expander("Change Password"):
        with st.form("change_password_form"):
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("Update Password"):
                if new_password and new_password == confirm_password:
                    st.session_state.app_password = new_password
                    st.success("Password updated successfully!")
                else:
                    st.error("Passwords do not match or are empty.")

# ============================================================
# PATIENT REGISTRATION
# ============================================================
elif page == "👤 Patient Registration":
    st.markdown('<div class="main-title">Patient AI Diagnosis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="P-1001")
            patient_name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=1, max_value=100, value=30)
        with col2:
            sex = st.selectbox("Biological Sex", ["female", "male"])
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            children = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        with col3:
            smoker = st.selectbox("Smoker Status", ["no", "yes"])
            region = st.selectbox("Geographic Region", ["southwest", "southeast", "northwest", "northeast"])
            notes = st.text_area("Physician Notes")
            
        submitted = st.form_submit_button("🔮 GENERATE AI PREDICTION", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if not patient_id.strip() or not patient_name.strip():
            st.error("⚠️ Please enter both Patient ID and Patient Name.")
        else:
            predicted = predict_charges(age, sex, bmi, children, smoker, region)
            cat, treatment = get_category_and_treatment(predicted)
            
            record = {
                "Patient ID": patient_id.strip(),
                "Patient Name": patient_name.strip(),
                "Age": age, "Gender": sex, "BMI": bmi, "Children": children,
                "Smoker": smoker, "Region": region,
                "Predicted Charges": round(predicted, 2),
                "Category": cat,
                "Treatment Type": treatment,
                "Doctor": st.session_state.doctor_name,
                "Date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
                "Notes": notes
            }
            st.session_state.history.append(record)
            st.session_state.last_prediction = record
            st.success("✅ Prediction generated successfully!")

    # SHOW RESULT
    if st.session_state.last_prediction:
        res = st.session_state.last_prediction
        
        # Gauge Chart for Price
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = res["Predicted Charges"],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Estimated Charges", 'font': {'size': 24}},
            number = {'prefix': "₹", 'valueformat': ",.2f"},
            gauge = {
                'axis': {'range': [None, 65000]},
                'bar': {'color': "#12355b"},
                'steps': [
                    {'range': [0, 5000], 'color': "#eefaf2"},
                    {'range': [5000, 15000], 'color': "#fff3cd"},
                    {'range': [15000, 30000], 'color': "#ffe5d0"},
                    {'range': [30000, 65000], 'color': "#f8d7da"}
                ]
            }
        ))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"<h2>Patient: {res['Patient Name']} ({res['Patient ID']})</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='treatment-badge'>{res['Treatment Type']}</div>", unsafe_allow_html=True)
            st.markdown(f"<h3>Risk Category: {res['Category']}</h3>", unsafe_allow_html=True)
            st.write(f"**Age:** {res['Age']} | **BMI:** {res['BMI']} | **Smoker:** {res['Smoker'].upper()}")
        with c2:
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PATIENT HISTORY
# ============================================================
elif page == "📋 Patient History":
    st.markdown('<div class="main-title">Patient Records Database</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No patient records available yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        search = st.text_input("🔎 Search by ID or Name")
        if search.strip():
            search_text = search.lower().strip()
            history_df = history_df[
                history_df["Patient ID"].str.lower().str.contains(search_text) |
                history_df["Patient Name"].str.lower().str.contains(search_text)
            ]
            
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Secure Records (CSV)", csv_data, "patient_history.csv", "text/csv", type="primary")

# ============================================================
# ADVANCED ANALYTICS
# ============================================================
elif page == "📊 Advanced Analytics":
    st.markdown('<div class="main-title">Healthcare Data Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌎 Demographics", "💰 Cost Drivers", "🩺 3D Analysis"])
    
    with tab1:
        st.markdown("### Demographic Cost Distribution")
        fig_sun = px.sunburst(df, path=['region', 'smoker', 'sex'], values='charges', 
                              title='Cost Breakdown by Region, Smoking Status, and Sex',
                              color='charges', color_continuous_scale='Viridis')
        st.plotly_chart(fig_sun, use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(df, x="smoker", y="charges", color="smoker", title="Impact of Smoking on Charges")
            st.plotly_chart(fig_box, use_container_width=True)
        with col2:
            fig_scatter = px.scatter(df, x="bmi", y="charges", color="smoker", size="age", 
                                     hover_data=["sex", "region"], title="BMI vs Charges (Size = Age)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    with tab3:
        st.markdown("### 3D Cost Landscape")
        fig_3d = px.scatter_3d(df, x='age', y='bmi', z='charges', color='smoker',
                               title='3D View: Age, BMI, and Charges',
                               opacity=0.7, size_max=10)
        st.plotly_chart(fig_3d, use_container_width=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "📈 AI Model Performance":
    st.markdown('<div class="main-title">AI Engine Diagnostics</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("We are using **Regression Models** and **EDA (Exploratory Data Analysis) techniques** in our project to estimate healthcare costs effectively and uncover underlying data patterns.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 R² Score (Accuracy)", f"{metrics['r2']:.4f}")
    col2.metric("📉 Mean Absolute Error (MAE)", f"₹ {metrics['mae']:,.2f}")
    col3.metric("📊 Root Mean Squared Error", f"₹ {metrics['rmse']:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Importance (Simulated extraction for pipeline)
    rf_model = model.named_steps["regressor"]
    cat_features = model.named_steps["preprocessor"].transformers_[0][1].get_feature_names_out(["sex", "smoker", "region"])
    num_features = ["age", "bmi", "children"]
    all_features = list(cat_features) + num_features
    
    importance = pd.DataFrame({
        "Feature": all_features,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig_imp = px.bar(importance, x="Importance", y="Feature", orientation='h', 
                     title="What drives healthcare costs? (Feature Importance)",
                     color="Importance", color_continuous_scale="Blues")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="footer"><b>🏥 MediCost Premium AI</b><br>Empowering doctors with next-generation predictive insights.<br>© 2026 MediCost Systems</div>', unsafe_allow_html=True)
