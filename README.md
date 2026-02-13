# 🏠 House Price Prediction (Streamlit App)

A modern **Machine Learning-based House Price Prediction** web application built with **Streamlit**.  
The app predicts property prices based on **Location, Area (sqft), BHK, and Bathrooms**, and presents results with confidence, recommendations, and a clean visual UI.

---

## ✨ Features
- 📍 Location-based price prediction in Gujarat  
- 🧠 Hybrid ML model (Linear Regression + Random Forest)  
- 📊 Price confidence & expected range  
- 💡 Smart investment recommendations  
- 🎨 Premium dark UI with interactive charts  

---

## 📁 Project Structure

C:.
│ .gitattributes
│ README.md
│ requirements.txt
│
+---app
│ app.py # Streamlit application
│
+---Dataset
│ gujarat_house_price_.csv # Dataset used for model training
│
+---Model
│ │ linear_model.pkl # Trained Linear Regression model
│ │ rf_model.pkl # Trained Random Forest model
│ │ location_encoder.pkl # Encoded location labels
│ │ model_train.py # Script to train ML models
│ │
│ +---.idea # IDE config files
│
---screenshots
Image-1.png # Application UI screenshots
Image-2.png
Image-3.png
Image-4.png
Image-5.png
Image-6.png
Image-7.png


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/House_Price_Prediction.git
cd House_Price_Prediction
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
streamlit run app/app.py
🧠 Model Details
Linear Regression (with StandardScaler)

Random Forest Regressor

Hybrid prediction using weighted average of both models

Log-transformed target variable for better accuracy