# Credit Scoring Web Application

A simple credit scoring web application built using Deep Learning and Explainable AI.  
The system predicts whether a customer is **High Risk** or **Low Risk** based on credit amount and provides visual explanations for the prediction.

---

## Tech Stack
- FastAPI
- PyTorch
- German Credit Dataset
- Scikit-learn
- Matplotlib
- Tailwind CSS

---

## Project Files
- `main.py` – Backend (FastAPI + ML + Explainability)
- `index.html` – Frontend UI
- `README.md` – Project documentation

---

## Setup & Run Instructions

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

---

### 3. Install Required Packages
```bash
pip install fastapi uvicorn torch pandas numpy scikit-learn matplotlib
```

---

### 4. Run the Application
```bash
uvicorn main:app --reload --port 9000
```

---

### 5. Open in Browser
```
http://127.0.0.1:9000
```

---

## How It Works
1. User enters credit amount
2. Model predicts credit risk score
3. Risk is classified as High or Low
4. Explanation charts show feature contributions

---

## Output
- Credit Risk Score
- Risk Classification (High / Low)
- Feature Importance Bar Chart
- Waterfall Explanation Diagram

---

## Note
This project uses a proxy target for demonstration purposes and is intended for academic and learning use.
