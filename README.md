
# 🧠 Multi-Disease Prediction System

A powerful AI-integrated healthcare assistant that predicts **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using machine learning, and provides **personalized health recommendations** powered by **Google Gemini AI**.

---

## 🚀 Features

* 🔍 **Disease Predictions** using trained ML models (`.sav` files):

  * Diabetes
  * Heart Disease
  * Parkinson's Disease
* 🤖 **Google Gemini AI Integration** for:

  * Health recommendations
  * Lifestyle guidance
  * Medication tips and diet suggestions
* 📊 **Risk Analysis**:

  * Radar comparison with average patient data
  * Trend graph for glucose-related diabetes risk
* 💬 **Health Chat Assistant**:

  * Ask any health-related question and get answers via Gemini AI

---

## 📦 Project Structure

```bash
├── diabetes_model.sav
├── heart_disease_model.sav
├── parkinsons_model.sav
├── disease.py
├── README.md
```

---

## 🧪 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit scikit-learn matplotlib numpy requests streamlit-option-menu
```

---

## 🧠 Gemini API Key Setup

You need a **Google Gemini API key** to use the health advice and chat features.
Get one from: [Google AI Studio](https://aistudio.google.com/app/apikey)

Replace your API key inside `disease.py`:

```python
API_KEY = "YOUR_API_KEY_HERE"
```

---

## ▶️ How to Run

1. **Navigate to your project folder:**

```bash
cd "Multiple Prediction System"
```

2. **Run the Streamlit App:**

```bash
streamlit run disease.py
```

---

## 🖼️ Preview

| Disease Prediction                     | AI Chat Assistant                  | Risk Graphs                         |
| -------------------------------------- | ---------------------------------- | ----------------------------------- |
| ![Diabetes](https://imgur.com/xyz.png) | ![Chat](https://imgur.com/abc.png) | ![Graph](https://imgur.com/def.png) |

---

## 🧑‍💻 Developed By

**Zohaib Shahid**
FAST NUCES Lahore Campus
[LinkedIn](https://www.linkedin.com/in/zohaib-shahid) | [GitHub](https://github.com/zohaib-7035)

---

## 📜 License

This project is for educational and non-commercial use. All health suggestions are AI-generated and **not** a substitute for professional medical advice.

---

