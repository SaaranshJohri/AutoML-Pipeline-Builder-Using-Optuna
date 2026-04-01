import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(path, target):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    target = target.lower()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    X = df.drop(columns=[target])
    y = df[target]

    # Encode target if categorical
    if y.dtype == "object" or str(y.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, "results/label_encoder.pkl")

    return train_test_split(X, y, test_size=0.2, random_state=42)