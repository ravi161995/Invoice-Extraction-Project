import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# ✅ Set dataset path
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# ✅ Set dataset path
import os
data_path = os.path.join(os.path.dirname(__file__), "data", "invoice_data.csv")

  # 🔁 update if your CSV name is different
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

# ✅ Load your data
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully:", df.shape)
print(df.head())

# 🧠 Replace column names below with your actual column names
text_column = "text"   # e.g. 'invoice_text' or 'description'
label_column = "label" # e.g. 'category' or 'type'

if text_column not in df.columns or label_column not in df.columns:
    raise KeyError(f"❌ Make sure your dataset has '{text_column}' and '{label_column}' columns")

texts = df[text_column].astype(str).values
labels = df[label_column].astype(str).values

# ✅ Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# ✅ TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
print("🚀 Training started...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
print("✅ Training completed!")

# ✅ Create save folder
save_dir = "E:/Invoice Extraction Project/app/models"
os.makedirs(save_dir, exist_ok=True)

# ✅ Save files
model.save(os.path.join(save_dir, "invoice_classifier.h5"))
joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

print(f"✅ All files saved successfully in {save_dir}")
  # 🔁 update if your CSV name is different
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

# ✅ Load your data
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully:", df.shape)
print(df.head())

# 🧠 Replace column names below with your actual column names
text_column = "text"   # e.g. 'invoice_text' or 'description'
label_column = "label" # e.g. 'category' or 'type'

if text_column not in df.columns or label_column not in df.columns:
    raise KeyError(f"❌ Make sure your dataset has '{text_column}' and '{label_column}' columns")

texts = df[text_column].astype(str).values
labels = df[label_column].astype(str).values

# ✅ Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# ✅ TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
print("🚀 Training started...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
print("✅ Training completed!")

# ✅ Create save folder
save_dir = "E:/Invoice Extraction Project/app/models"
os.makedirs(save_dir, exist_ok=True)

# ✅ Save files
model.save(os.path.join(save_dir, "invoice_classifier.h5"))
joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

print(f"✅ All files saved successfully in {save_dir}")
