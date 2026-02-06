import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.set_page_config(page_title="ML Model Comparison App", layout="wide")

st.title("📊 Machine Learning Classification App")
st.write("Upload test data, select a model, and view predictions & metrics.")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/saved_models/lr.pkl"),
        "Decision Tree": joblib.load("model/saved_models/dt.pkl"),
        "KNN": joblib.load("model/saved_models/knn.pkl"),
        "Naive Bayes": joblib.load("model/saved_models/gnb.pkl"),
        "Random Forest": joblib.load("model/saved_models/rf.pkl"),
        "XGBoost": joblib.load("model/saved_models/xgb.pkl"),
    }

def load_label_encoder():
    return joblib.load("model/saved_models/label_encoder.pkl")

def getMetrics(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = compute_auc(model, X_test, y_test)
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test, y_pred,average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    numeric_labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=numeric_labels)

    return {'Accuracy':round(accuracy,4), 'AUC':("NA" if(auc==None) else round(auc,4)),'Precision':round(precision, 4),
            'Recall': round(recall,4),'F1-Score': round(f1,4), 'Matthews Correlation Coefficient': round(mcc, 4),'Confusion Matrix': cm,
            'CM Labels': numeric_labels}

def compute_auc(model, X_test, y_test):
    # AUC needs probabilities
    if not hasattr(model, "predict_proba"):
        return None

    y_test = np.array(y_test)
    unique_classes = np.unique(y_test)

    # AUC undefined if less than 2 classes in y_test
    if len(unique_classes) < 2:
        return None

    y_prob = model.predict_proba(X_test)

    # Binary classification
    if y_prob.shape[1] == 2:
        return roc_auc_score(y_test, y_prob[:, 1])

    # Multiclass: STRICT shape check
    if y_prob.shape[1] != len(unique_classes):
        return None

    return roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

models = load_models()

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# -------------------------------
# Main logic
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "NObeyesdad" not in df.columns:
        st.error("❌ Dataset must contain a column named 'NObeyesdad'")
    else:
        X_test = df.drop(columns=["NObeyesdad"])
        label_encoder = load_label_encoder()
        y_test = df["NObeyesdad"] if(model_name!='XGBoost') else label_encoder.transform(df["NObeyesdad"])

        model = models[model_name]
        labels = model.named_steps['classifier'].classes_ if(model_name!='XGBoost') else label_encoder.classes_
        print(labels)
        metrics = getMetrics(model, X_test, y_test)

        st.subheader(f"📈 Evaluation Metrics — {model_name}")

        col1, col2, col3 = st.columns(3)

        for m in metrics.keys():
            if(m not in ['Confusion Matrix', 'CM Labels']):
                col1.metric(m, metrics[m])

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        st.subheader("🧩 Confusion Matrix")

        cm = metrics['Confusion Matrix']
        cm_labels = metrics['CM Labels']
        
        # Map numeric → string ONLY for display
        if model_name == "XGBoost":
            display_labels = [label_encoder.classes_[i] for i in cm_labels]
        else:
            display_labels = cm_labels
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        )
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
