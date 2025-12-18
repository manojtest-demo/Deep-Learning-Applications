import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# -----------------------------
# Load dataset
# -----------------------------
url = "https://raw.githubusercontent.com/manojtest-demo/Deep-Learning-Applications/main/german_credit_data.csv"
df = pd.read_csv(url)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Simple proxy target
df["default"] = (df["Credit amount"] > df["Credit amount"].median()).astype(int)

y = df["default"]
X = pd.get_dummies(df.drop("default", axis=1), drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

# -----------------------------
# Model (Simple & Stable)
# -----------------------------
class CreditModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = CreditModel(X_t.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for _ in range(15):
    optimizer.zero_grad()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    optimizer.step()

model.eval()

# -----------------------------
# API
# -----------------------------
class CreditInput(BaseModel):
    features: dict

@app.get("/")
def home():
    return FileResponse("index.html")

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.post("/predict")
def predict(data: CreditInput):

    input_df = pd.DataFrame([data.features])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    input_t = torch.tensor(input_scaled, dtype=torch.float32, requires_grad=True)
    score = model(input_t)
    score.backward()

    # -----------------------------
    # SHAP-style explanation (FAST)
    # -----------------------------
    grads = input_t.grad.numpy()[0]
    contrib = grads * input_scaled[0]

    abs_vals = np.abs(contrib)
    idx = np.argsort(abs_vals)[::-1][:5]

    features = np.array(X.columns)[idx]
    values = abs_vals[idx]
    directions = contrib[idx]

    colors = ["#dc2626" if v > 0 else "#16a34a" for v in directions]

    plt.figure(figsize=(7, 4))
    plt.barh(features, values, color=colors)
    plt.xlabel("Impact on Credit Risk")
    plt.title("Top Factors Influencing Credit Risk")
    plt.gca().invert_yaxis()

    explanation_bar = fig_to_base64()

    # -----------------------------
    # Risk Confidence Level (Nice touch)
    # -----------------------------
    s = score.item()
    if s < 0.35:
        confidence = "Low Risk"
    elif s < 0.65:
        confidence = "Medium Risk"
    else:
        confidence = "High Risk"

    return JSONResponse({
        "credit_score": round(s, 3),
        "risk": "High Risk" if s > 0.5 else "Low Risk",
        "confidence_level": confidence,
        "explanation_bar": explanation_bar
    })
