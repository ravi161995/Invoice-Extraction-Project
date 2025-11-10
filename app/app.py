import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import easyocr
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import cv2


# ------------------------------------------------------------
# 1️⃣ Streamlit Page Configuration (MUST be first Streamlit command)
# ------------------------------------------------------------
st.set_page_config(page_title="Invoice Information Extractor", layout="wide")

# ------------------------------------------------------------
# 2️⃣ Load Model, Vectorizer, and Label Encoder
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_tools():
    base_path = os.path.dirname(__file__)
    model_dir = os.path.join(base_path, "models")

    model_path = os.path.join(model_dir, "invoice_classifier.h5")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
        st.error("❌ Model files not found! Please make sure training has been completed.")
        st.stop()

    model = tf.keras.models.load_model(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, vectorizer, label_encoder


# ------------------------------------------------------------
# 3️⃣ Load EasyOCR Reader (with Permission Error Fix)
# ------------------------------------------------------------
@st.cache_resource
def load_easyocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except PermissionError:
        # Delete temp.zip if locked
        temp_zip = os.path.expanduser("~/.EasyOCR/model/temp.zip")
        if os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except Exception:
                pass
        return easyocr.Reader(['en'], gpu=False)


# ------------------------------------------------------------
# 4️⃣ Load Everything
# ------------------------------------------------------------
model, vectorizer, label_encoder = load_model_and_tools()
reader = load_easyocr_reader()

# ------------------------------------------------------------
# 5️⃣ Streamlit UI Layout
# ------------------------------------------------------------
st.title("📄 Invoice Information Extractor")
st.write("Upload an invoice image, and this app will extract key information using OCR and AI model.")

uploaded_file = st.file_uploader("📤 Upload Invoice Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="🧾 Uploaded Invoice", use_column_width=True)

    # Run OCR
    with st.spinner("🔍 Extracting text using OCR..."):
        results = reader.readtext(img_array)
        extracted_text = [res[1] for res in results]
        joined_text = " ".join(extracted_text)

    st.subheader("📋 Extracted Text")
    st.write(joined_text)

    # Predict with model
    with st.spinner("🤖 Classifying extracted text..."):
        X_vec = vectorizer.transform([joined_text])
        prediction = model.predict(X_vec)
        predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

    st.success(f"🏷️ Predicted Label: **{predicted_label}**")

else:
    st.info("👆 Please upload an invoice image to begin.")

# ------------------------------------------------------------
# 6️⃣ Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("🧠 Developed by Ravi | AI Invoice Extraction Project")
