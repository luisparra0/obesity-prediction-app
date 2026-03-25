#  Obesity Risk Prediction App

Machine Learning application for preventive obesity risk assessment, built with XGBoost and deployed through an interactive Streamlit dashboard.

The system allows healthcare professionals to evaluate patient risk, track history, and explore data insights in a simple and intuitive interface.

---

##  Application Preview

###  Home
![Home](assets/images/home.png)

###  Prediction (Triage Form)
![Prediction](assets/images/predict.png)

###  Patient History
![History](assets/images/history.png)

###  Dashboard (EDA)
![Dashboard](assets/images/dashboards.png)

---

##  Business Value

- Supports early identification of obesity risk  
- Helps healthcare professionals with decision-making  
- Generates automated patient risk evaluations  
- Enables historical tracking of patient records  
- Provides data-driven insights through dashboards  

---

## ⚙️ How to Run

```bash
git clone https://github.com/luisparra0/obesity-prediction-app.git
cd obesity-prediction-app

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run src/app.py
```

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib  
- Seaborn  

---

##  Model

- Algorithm: XGBoost Classifier  
- Task: Obesity Risk Classification  
- Performance (Notebook):
  - Accuracy ≈ 96%  
  - Cross-validation applied  

---

##  Project Structure

```bash
obesity-prediction-app/

├── src/
│   ├── app.py
│   ├── pages/
│   ├── models/
│   └── shared/
│
├── data/
├── notebook/
├── assets/
│   └── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Features

- Interactive triage form  
- Real-time ML prediction  
- Exploratory data analysis (EDA)  
- Patient history tracking (SQLite)  
- CSV export functionality  
- Clean and intuitive UI  

---

##  How It Works

1. User fills in patient information in the Predict tab  
2. The trained XGBoost model generates a prediction  
3. The result is stored in a local database  
4. Data can be explored in:
   - History tab  
   - Dashboard (EDA)  

---

## ⚠️ Notes

- The app uses a locally trained model (.joblib)  
- Metrics were removed from the app to avoid inconsistencies  
- Model evaluation is available in the notebook (analytics.ipynb)  

---

##  Author

Luís Parra  
Data Analyst focused on Machine Learning and Data Products
