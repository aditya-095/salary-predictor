import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Color and Style Tokens ---
BG_GRAD = "linear-gradient(120deg,#212336 15%,#1a1e30 100%)"
SIDEBAR_GRAD = "linear-gradient(135deg,#191e32 85%,#303e5c 100%)"
GLASS = "rgba(28, 33, 53, 0.95)"
ACCENT = "#4fdbb4"
BUTTON_GRAD = "linear-gradient(90deg, #32dbc6 0%, #507dff 100%)"
HOVER_GRAD = "linear-gradient(90deg,#507dff 0%, #32dbc6 100%)"
DANGER = "#e96f78"
MAIN_TXT = "#fcfcfc"
SECOND_TXT = "#d4f3fc"

st.set_page_config(
    page_title="Employee Salary Predicting System",
    page_icon="✨",
    layout="wide"
)

# --- Custom CSS ---
st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Manrope', 'Segoe UI', Arial, sans-serif!important;
        background: {BG_GRAD} !important;
        color: {MAIN_TXT};
    }}
    .stApp {{
        background: {BG_GRAD} !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        background: {SIDEBAR_GRAD};
        border-radius: 0 18px 18px 0;
        padding-top: 28px;
        min-width: 275px;
        box-shadow: 8px 0 22px #151e3338 !important;
    }}
    .sidebar-title {{
        color: {ACCENT};
        font-size: 1.25em; font-weight: bold; margin-bottom: 6px;
    }}
    .sidebar-inst {{
        color: {SECOND_TXT};
        font-size: 1.07em;
        margin-bottom: 19px;
        line-height:1.7;
    }}
    .glass-card {{
        background: {GLASS};
        border-radius: 18px;
        box-shadow: 0 4px 28px #181c2436;
        -webkit-backdrop-filter: blur(24px);
        backdrop-filter: blur(12px);
        padding: 2.1em 2em 1.24em 2em;
        margin-bottom: 1.8em;
    }}
    .result-card {{
        background: rgba(36,40,66,0.97);
        border-radius: 18px;
        padding: 2.5em 2.05em;
        box-shadow: 0 5px 22px #141c2442;
    }}
    .mainhead {{
        color: {ACCENT};
        font-size: 2.16em;
        font-weight: 800;
        letter-spacing:1.6px;
    }}
    .subtxt {{
        color: #bdfce7;
        font-size: 1.09em;
        margin-bottom: 1.15em;
        font-weight: 600;
    }}
    .new-predict-btn > button {{
        background: {BUTTON_GRAD} !important;
        color: #212336 !important;
        border: none;
        border-radius: 12px !important;
        padding: 0.56em 2.66em !important;
        margin-top: 0.7em !important;
        font-size: 1.16em !important;
        font-weight: 900 !important;
        box-shadow: 0 2px 18px #4fdbb478;
        transition: background 0.2s, color 0.25s, box-shadow 0.19s;
        letter-spacing: .2px;
    }}
    .new-predict-btn > button:hover {{
        background: {HOVER_GRAD} !important;
        color: #ffffff !important;
        box-shadow: 0 2px 36px #81e0cf29;
    }}
    hr {{
        border: none; border-top: 2px solid #222b;
        margin-top: 2em; margin-bottom: 1.5em;
    }}
    .stDataFrame thead tr th {{
        background: #272940 !important; color: {ACCENT} !important;
        font-weight: 700 !important;
    }}
    .stDataFrame tbody tr td {{
        background: #182639 !important; color: {SECOND_TXT};
    }}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Dashboard: How to Use</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-inst'>"
        "<ul>"
        "<li>Fill in <b>all profile and work fields</b>.</li>"
        "<li>Click <span style='color:#4fdbb4;font-weight:600;'>Predict Salary</span>.</li>"
        "<li>Review income category, salary, and confidence in the cards shown.</li>"
        "<li>Expand details for more insights.</li>"
        "</ul>"
        "<span style='color:#8becc9;font-size:0.98em;font-weight:600;'>You can collapse this sidebar from the ☰ icon above.</span>"
        "</div>",
        unsafe_allow_html=True
    )

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        base_path = os.path.dirname(__file__)
        clf_model = joblib.load(os.path.join(base_path, "salary_classification_model.pkl"))
        reg_model = joblib.load(os.path.join(base_path, "salary_regression_model.pkl"))
        label_encoder = joblib.load(os.path.join(base_path, "label_encoder.pkl"))
        column_info = joblib.load(os.path.join(base_path, "column_info.pkl"))
        return clf_model, reg_model, label_encoder, column_info
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}")
        st.stop()

clf_model, reg_model, label_encoder, column_info = load_models()

# --- UI Input ---
st.markdown("<div class='mainhead'>Employee Salary Predicting System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtxt'>Instant AI salary insights for every workplace profile. Enter your data — estimate your income potential.</div><hr>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### Profile Input")
    user_input = {}

    ci = column_info['all_columns'] if 'all_columns' in column_info else []
    rows = [st.columns(4), st.columns(4), st.columns(4)]

    if 'age' in ci:
        user_input['age'] = rows[0][0].number_input("Age", 16, 90, 30)
    if 'gender' in ci:
        user_input['gender'] = rows[0][1].selectbox("Gender", ["Male", "Female"])
    elif 'sex' in ci:
        user_input['sex'] = rows[0][1].selectbox("Gender", ["Male", "Female"])
    if 'race' in ci:
        user_input['race'] = rows[0][2].selectbox("Race", [
            "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    if 'native-country' in ci:
        user_input['native-country'] = rows[0][3].selectbox("Country", [
            "United-States", "Mexico", "Philippines", "Germany", "Puerto-Rico", "Canada", "El-Salvador", "India", "Cuba", "England", "China", "Other"])

    if 'workclass' in ci:
        user_input['workclass'] = rows[1][0].selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    if 'education' in ci:
        user_input['education'] = rows[1][1].selectbox("Education Level", [
            "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
    if 'educational-num' in ci:
        user_input['educational-num'] = rows[1][2].slider("Education Years", 1, 16, 10)
    elif 'education-num' in ci:
        user_input['education-num'] = rows[1][2].slider("Education Years", 1, 16, 10)
    if 'occupation' in ci:
        user_input['occupation'] = rows[1][3].selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
            "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
            "Priv-house-serv", "Protective-serv", "Armed-Forces"])

    if 'hours-per-week' in ci:
        user_input['hours-per-week'] = rows[2][0].slider("Hours/Week", 1, 100, 40)
    if 'marital-status' in ci:
        user_input['marital-status'] = rows[2][1].selectbox("Marital Status", [
            "Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"])
    if 'relationship' in ci:
        user_input['relationship'] = rows[2][2].selectbox("Relationship", [
            "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
    if 'capital-gain' in ci:
        user_input['capital-gain'] = rows[2][3].number_input("Capital Gain", 0, 100000, 0)
    if 'capital-loss' in ci:
        st.columns([1, 1, 1, 1])[0].number_input("Capital Loss", 0, 10000, 0, key="caploss")
        user_input['capital-loss'] = st.session_state.caploss

    st.markdown("</div>", unsafe_allow_html=True)

# --- Predict Button ---
center2 = st.columns([5, 1.2, 5])
with center2[1]:
    predict = st.container()
    with predict:
        predicted = st.button("Predict Salary", key="predict-btn", use_container_width=True)
    predict.markdown('<div style="margin-top:-2.4em"></div>', unsafe_allow_html=True)

# --- Results ---
if 'predicted' in locals() and predicted:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("#### Salary Results")
    try:
        input_df = pd.DataFrame([user_input])
        for col in ci:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[ci]

        class_prediction = clf_model.predict(input_df)[0]
        class_prob = clf_model.predict_proba(input_df)[0]
        predicted_category = label_encoder.inverse_transform([class_prediction])[0]
        salary_prediction = reg_model.predict(input_df)[0]

        st.markdown(
            f"<div style='display:flex;gap:4vw;flex-wrap:wrap;margin-bottom:1.05em;'>"
            f"<div style='flex:1;min-width:197px'><div style='color:{ACCENT};font-size:1.22em;font-weight:800;'>Income Category</div>"
            f"<div style='font-size:2.18em;color:{(DANGER if '<=50K' in predicted_category else ACCENT)};padding-top:0.15em; font-weight:900;'>{predicted_category}</div>"
            f"<div style='font-size:1.03em;color:{SECOND_TXT};margin:0.67em 0 0.56em 0;'>Confidence: {max(class_prob)*100:.1f}%</div></div>"
            f"<div style='flex:1;min-width:196px'><div style='color:{ACCENT};font-size:1.22em;font-weight:800;'>Estimated Salary</div>"
            f"<div style='font-size:2.03em;color:{MAIN_TXT};font-weight:900;'>${salary_prediction:,.0f}</div>"
            f"<div style='color:{SECOND_TXT};font-size:0.96em'>Annual Estimate</div></div>"
            f"</div>",
            unsafe_allow_html=True
        )
        with st.expander("Show Probability Details", expanded=False):
            prob_df = pd.DataFrame({
                'Category': label_encoder.classes_,
                'Probability': class_prob
            }).sort_values('Probability', ascending=False)
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
        if "<=50K" in predicted_category:
            st.warning("This profile is predicted in the lower income group.")
        else:
            st.success("This profile aligns with a higher earner group.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Please check all fields and try again.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:left;color:{SECOND_TXT};margin-top:8px;font-size:1.04em;'>"
    "Employee Salary Predicting System &ndash; AI-powered salary analytics.<br>Powered by Streamlit & Scikit-learn."
    "<br><span style='color:#8becc9;'>Results are predictive only.</span></div>",
    unsafe_allow_html=True
)

