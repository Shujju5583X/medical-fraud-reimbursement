# 🏥 AcuClaim: AI-Powered Medical Reimbursement and Fraud Detection System

This project is an **end-to-end machine learning application** that detects fraudulent medical reimbursement claims. It includes a **FastAPI backend**, **HTML/CSS/JS frontend**, and an **ML model** for predictions.

---

## 🚀 Features

✅ Upload medical claim data for fraud detection
✅ Real-time prediction using ML model
✅ User-friendly HTML + CSS frontend
✅ FastAPI backend for model inference
✅ Scalable design for bulk predictions

---

## 📂 Project Structure

```
medical-fraud-reimbursement/
│
├── backend/
│   ├── app.py              # FastAPI app
│   ├── model.pkl           # Trained ML model
│   ├── requirements.txt    # Python dependencies
│
├── frontend/
│   ├── index.html          # Frontend UI
│   ├── style.css           # Styling
│   ├── script.js           # Handles API calls
│
└── README.md
```

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML, CSS, JavaScript
* **ML:** Scikit-learn, Pandas, Joblib
* **Others:** Jinja2 (for template rendering)

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/medical-fraud-reimbursement.git
cd medical-fraud-reimbursement
```

### 2️⃣ Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4️⃣ Run the Backend Server

```bash
cd backend
uvicorn app:app --reload
```

**Server will run on:** `http://127.0.0.1:8000`

---

## 🌐 Running the Frontend

Open `frontend/index.html` in your browser or serve it using:

```bash
python -m http.server 5500
```

**Access via:** `http://127.0.0.1:5500/frontend/index.html`

---

## 📡 API Endpoints

* **` POST /predict`** → Predict fraud for single record
* **` POST /predict_bulk`** → Predict fraud for multiple records (CSV upload)
* **`GET /`** → Frontend page

## ✅ To Do

* [ ] Add user authentication
* [ ] Improve UI design
* [ ] Deploy on cloud (Render / AWS / Heroku)
