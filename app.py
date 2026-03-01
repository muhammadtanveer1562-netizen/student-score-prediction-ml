import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Academic Intelligence",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("best_model.pkl")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🚀 AI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📊 Analytics", "ℹ️ About System"])

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30, #243B55);
}
.big-title {
    font-size:40px;
    font-weight:bold;
    color:white;
}
.subtitle {
    font-size:18px;
    opacity:0.7;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ===================== DASHBOARD =====================
if page == "🏠 Dashboard":

    # -------- HEADER WITH IMAGE --------
    colA, colB = st.columns([1, 2])

    with colA:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
            width=220
        )

    with colB:
        st.markdown("<div class='big-title'>AI Student Performance Engine</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Enterprise Academic Intelligence System</div>", unsafe_allow_html=True)

    st.write("")

    # -------- INPUT SECTION --------
    col1, col2, col3 = st.columns(3)

    study_hours = col1.slider("Study Hours", 0.0, 12.0, 4.0)
    attendance = col2.slider("Attendance %", 0.0, 100.0, 85.0)
    mental_health = col3.slider("Mental Health", 1, 10, 6)

    sleep_hours = col1.slider("Sleep Hours", 0.0, 12.0, 7.0)
    part_time_job = col2.toggle("Part-Time Job")

    ptj_encoded = 1 if part_time_job else 0

    # -------- PREDICTION --------
    if st.button("⚡ Run AI Model"):

        with st.spinner("Running Deep Academic Analysis..."):
            time.sleep(1)

            input_data = np.array([[study_hours,
                                    attendance,
                                    mental_health,
                                    sleep_hours,
                                    ptj_encoded]])

            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))

        # -------- GAUGE CHART --------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'suffix': "%"},
            title={'text': "Predicted Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00f5d4"},
                'steps': [
                    {'range': [0, 50], 'color': "#ff4d6d"},
                    {'range': [50, 70], 'color': "#ffd166"},
                    {'range': [70, 85], 'color': "#06d6a0"},
                    {'range': [85, 100], 'color': "#00f5d4"}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="#141E30",
            font={'color': "white"},
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # KPI CARDS
        k1, k2, k3 = st.columns(3)
        k1.metric("Study Hours", study_hours)
        k2.metric("Attendance", f"{attendance}%")
        k3.metric("Mental Health", mental_health)

# ===================== ANALYTICS =====================
elif page == "📊 Analytics":

    st.markdown("<h1 style='color:white;'>Feature Impact Analysis</h1>", unsafe_allow_html=True)

    features = {
        "Study Hours": 35,
        "Attendance": 30,
        "Mental Health": 15,
        "Sleep": 12,
        "Part-Time Job": 8
    }

    fig = px.bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation='h',
        title="Model Feature Importance",
        color=list(features.values()),
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        plot_bgcolor="#141E30",
        paper_bgcolor="#141E30",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===================== ABOUT =====================
else:

    st.markdown("<h1 style='color:white;'>About This System</h1>", unsafe_allow_html=True)

    st.write("""
    ### 🔬 AI Academic Intelligence System

    This enterprise-grade application predicts student performance 
    using Machine Learning techniques.

    **Core Technologies:**
    - Python
    - Streamlit
    - Scikit-Learn
    - Plotly Visualization
    
    **Capabilities:**
    - Real-time prediction
    - Interactive gauge analytics
    - Feature impact visualization
    - Enterprise dashboard UI


    Developed by Tanveer Khan 🚀
    """)
#python -m streamlit run app.py