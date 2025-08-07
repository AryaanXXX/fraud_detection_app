import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay, precision_recall_curve,
                             PrecisionRecallDisplay)
from imblearn.over_sampling import SMOTE
import shap
import joblib
import os
import io
import requests




# --- Configuration & Setup ---
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🔐 Credit Card Fraud Detection App")

DATA_FILE = "creditcard.csv"
MODEL_FILE = "fraud_model.joblib"
SCALER_FILE = "scaler.joblib"




# --- Session State Management ---
if 'model_trained' not in st.session_state:
    st.session_state.update({
        'model_trained': False,
        'model': None,
        'model_choice': None,
        'threshold': 0.5,
        'data_loaded': False,
        'df': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'scaler': None,
        'y_proba': None,
        'y_pred': None
    })





@st.cache_data
def load_data_from_url(url):
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df

CSV_URL = "https://drive.google.com/uc?export=download&id=1-AmrZLwBvrtMa0KbnwHCyyoX88sK-bBa"

df = load_data_from_url(CSV_URL)






# --- Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    """Load data, handle missing files, and perform preprocessing (scaling & SMOTE)."""
    try:
        if not os.path.exists(DATA_FILE):
            st.error(f"Dataset file '{DATA_FILE}' not found. Please ensure it's in the same directory.")
            return None  # Return None on error

        df = pd.read_csv(DATA_FILE)
        required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            st.error(f"Missing required columns in dataset: {missing}")
            return None  # Return None on error

        X = df.drop(['Class', 'Time'], axis=1)
        y = df['Class']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        return df, X_train_res, y_train_res, X_test, y_test, scaler

    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None  # Return None on any other error





# Load and preprocess data on first run
if not st.session_state.data_loaded:
    with st.spinner("Loading and preprocessing data..."):
        # Corrected block to handle the potential NoneType return
        data_tuple = load_and_preprocess_data()
        if data_tuple is not None:
            st.session_state.df, st.session_state.X_train, st.session_state.y_train, \
                st.session_state.X_test, st.session_state.y_test, st.session_state.scaler = data_tuple
            st.session_state.data_loaded = True
        else:
            st.stop()  # Stop the app gracefully if data loading failed




# --- EDA Section ---
st.header("1. Dataset Overview")
with st.expander("View Data and Summary"):
    st.write(st.session_state.df.head(3))
    st.write(f"Shape: {st.session_state.df.shape}")
    st.write("Missing Values:", st.session_state.df.isnull().sum().sum())

st.subheader("Class Distribution")
legit, fraud = st.session_state.df['Class'].value_counts()
col1, col2 = st.columns(2)
with col1:
    st.metric("Legitimate Transactions", f"{legit:,}")
    st.metric("Fraudulent Transactions", f"{fraud:,}", delta=f"{fraud / (legit + fraud):.2%} of total")
with col2:
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=st.session_state.df, ax=ax)
    st.pyplot(fig)




# --- Model Training & Loading ---
st.header("2. Model Training")
st.markdown("Choose a model and train it. The trained model and scaler will be saved locally for future use.")
model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost", "Ensemble"],
    key="model_select"
)




# Load existing model if available on app restart
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and not st.session_state.model_trained:
    with st.spinner("Loading saved model and scaler..."):
        try:
            st.session_state.model = joblib.load(MODEL_FILE)
            st.session_state.scaler = joblib.load(SCALER_FILE)
            st.session_state.model_trained = True

            # This handles the model choice not being set from a loaded joblib
            if isinstance(st.session_state.model, RandomForestClassifier):
                st.session_state.model_choice = "Random Forest"
            elif isinstance(st.session_state.model, XGBClassifier):
                st.session_state.model_choice = "XGBoost"
            else:
                st.session_state.model_choice = "Ensemble"

            st.session_state.y_proba = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
            st.session_state.y_pred = (st.session_state.y_proba >= st.session_state.threshold).astype(int)
        except Exception as e:
            st.warning(f"Could not calculate metrics for loaded model: {e}")
            st.session_state.model_trained = False


if st.button("Train Model" if not st.session_state.model_trained else "Retrain Model", type="primary", key="train_btn"):
    with st.status("Training in progress...", expanded=True) as status:
        try:
            y_train_res = st.session_state.y_train
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
            elif model_choice == "XGBoost":
                scale_pos_weight = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])
                model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='aucpr', random_state=42)
            else:  # Ensemble
                scale_pos_weight = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])
                model = VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                        ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42))
                    ],
                    voting='soft'
                )

            model.fit(st.session_state.X_train, st.session_state.y_train)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(st.session_state.scaler, SCALER_FILE)

            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.model_choice = model_choice

            st.session_state.y_proba = model.predict_proba(st.session_state.X_test)[:, 1]
            st.session_state.y_pred = (st.session_state.y_proba >= st.session_state.threshold).astype(int)

            status.update(label="Training complete!", state="complete")
            st.success("Model trained and saved successfully!")

            st.rerun()

        except Exception as e:
            status.update(label="Training failed", state="error")
            st.error(f"Training Error: {str(e)}")





# --- Model Evaluation ---
if st.session_state.model_trained:
    st.header("3. Model Evaluation")

    st.subheader("Threshold Tuning")
    st.session_state.threshold = st.slider(
        "Set fraud probability threshold",
        min_value=0.01,
        max_value=0.99,
        value=st.session_state.threshold,
        step=0.01,
        help="Higher values reduce false positives but may miss more fraud"
    )

    st.session_state.y_pred = (st.session_state.y_proba >= st.session_state.threshold).astype(int)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Legit', 'Actual Fraud'], columns=['Predicted Legit', 'Predicted Fraud'])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax)
    plt.title(f"Threshold = {st.session_state.threshold:.2f}")
    st.pyplot(fig)

    st.subheader("Performance Metrics")
    tn, fp, fn, tp = cm.ravel()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Detection Rate (Recall)", f"{tp / (tp + fn):.1%}" if (tp + fn) > 0 else "N/A",
                  help="Percentage of actual fraud cases detected")
    with col2:
        st.metric("False Positive Rate", f"{fp / (fp + tn):.1%}" if (fp + tn) > 0 else "N/A",
                  help="Percentage of legit transactions flagged as fraud")
    with col3:
        st.metric("Precision", f"{tp / (tp + fp):.1%}" if (tp + fp) > 0 else "N/A",
                  help="Percentage of fraud predictions that are correct")

    with st.expander("Detailed Classification Report"):
        st.text(classification_report(st.session_state.y_test, st.session_state.y_pred,
                                      target_names=["Legitimate", "Fraud"]))

    st.subheader("Performance Curves")
    tab1, tab2 = st.tabs(["ROC Curve", "Precision-Recall"])

    with tab1:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(st.session_state.model, st.session_state.X_test, st.session_state.y_test, ax=ax)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(st.session_state.model, st.session_state.X_test, st.session_state.y_test,
                                              ax=ax)
        st.pyplot(fig)




    # --- Feature Importance (SHAP) ---
    st.header("4. Feature Importance (SHAP)")
    try:
        X_sample = pd.DataFrame(st.session_state.X_test[:500],
                                columns=st.session_state.df.drop(['Class', 'Time'], axis=1).columns)

        model_to_explain = None
        if st.session_state.model_choice == "Ensemble":
            model_to_explain = st.session_state.model.named_estimators_['xgb']
        else:
            model_to_explain = st.session_state.model

        explainer = shap.TreeExplainer(model_to_explain)
        shap_values = explainer(X_sample)

        st.subheader("Top Fraud Indicators")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP visualization unavailable: {str(e)}")





# --- Prediction Interface ---
st.header("5. Make Predictions on New Data")
uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"],
                                 help="Upload a CSV with the same columns (excluding 'Time' and 'Class') to get fraud predictions.")

if uploaded_file and st.session_state.model_trained:
    try:
        new_data_df = pd.read_csv(uploaded_file)

        required_cols = st.session_state.df.drop(['Class', 'Time'], axis=1).columns.tolist()
        if not all(col in new_data_df.columns for col in required_cols):
            missing_cols = set(required_cols) - set(new_data_df.columns)
            st.error(f"The uploaded file is missing the following required columns: {missing_cols}")
            st.stop()

        with st.spinner("Making predictions..."):
            new_data_scaled = st.session_state.scaler.transform(new_data_df[required_cols])

            try:
                model_to_predict = st.session_state.model
                if st.session_state.model_choice == "Ensemble":
                    model_to_predict = st.session_state.model.named_estimators_['xgb']

                probas = model_to_predict.predict_proba(new_data_scaled)[:, 1]
                preds = (probas >= st.session_state.threshold).astype(int)

            except Exception as e:
                st.error(f"Prediction failed. Error during model inference: {e}")
                st.stop()

        results_df = new_data_df.copy()
        results_df['Fraud_Probability'] = probas
        results_df['Prediction'] = ['Fraud' if p == 1 else 'Legitimate' for p in preds]

        st.subheader("Prediction Results")
        st.info(f"Showing results for {len(results_df):,} transactions.")

        display_df = results_df.head(50) if len(results_df) > 50 else results_df


        def color_fraud(val):
            return 'color: red; font-weight: bold;' if val == 'Fraud' else ''


        st.dataframe(
            display_df.style.applymap(color_fraud, subset=['Prediction']).format({'Fraud_Probability': '{:.2%}'})
        )

        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Full Results",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()

elif uploaded_file and not st.session_state.model_trained:
    st.warning("Please train a model first before making predictions.")




# --- Footer ---
st.markdown("---")
st.caption("© 2025 Fraud Detection System | v2.7")





#how to access this code or run the code...
#  cd /Users/asadullahibnferdous/Documents/AI\ and\ ML/
#  streamlit run fraud_detection.py


















