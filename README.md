# Bengaluru House Price Prediction 🏠

A machine learning project to predict Bengaluru house prices using Python, Pandas, Scikit-learn, Random Forest, and Streamlit.

## 📌 Project Overview

This project predicts the estimated price of a house in Bengaluru based on details such as location, total square footage, number of bedrooms, and bathrooms.

The project uses a **Random Forest Regressor** machine learning model and a Streamlit web application to make house price predictions through a simple user interface.

## 🎯 Project Objectives

- Understand and preprocess real-world housing data.
- Build a machine learning model using Random Forest Regression.
- Estimate house prices based on user inputs.
- Create an interactive web application using Streamlit.
- Practice Python, machine learning, and project deployment skills.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine learning model training and evaluation
- **Random Forest Regressor** – House price prediction algorithm
- **Streamlit** – Web application interface
- **Git and GitHub** – Version control and project hosting

## 📂 Project Structure

```text
house_price_prediction/
│
├── data/
│   └── Bengaluru_House_Data.csv
│
├── app.py
├── house_price_utils.py
├── train_house_price_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

The project uses Bengaluru housing data containing property-related information, including location, square footage, number of bedrooms, bathrooms, and price.

The dataset is stored in the `data/` directory.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pavithra-web68/Bengaluru-house-price-prediction.git
```

### 2. Open the project folder

```bash
cd Bengaluru-house-price-prediction
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

### 4. Activate the virtual environment on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

### Step 1: Train the machine learning model

```bash
python train_house_price_model.py
```

### Step 2: Start the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

**Note:** If the training script saves a model file, follow the output and file paths defined in the project code.

## 🧠 Machine Learning Workflow

1. Load the housing dataset.
2. Explore and clean the data.
3. Prepare the input features and target variable.
4. Train a Random Forest Regression model.
5. Evaluate model performance.
6. Use the trained model to estimate house prices.
7. Display predictions through a Streamlit application.

## 🌱 Learning Outcomes

- Working with a real-world dataset.
- Performing basic data preprocessing.
- Understanding supervised machine learning.
- Training and evaluating a regression model.
- Building an interactive application with Streamlit.
- Managing a project using Git and GitHub.

## 🔮 Future Improvements

- Improve data cleaning and feature engineering.
- Compare Random Forest with other regression algorithms.
- Improve model accuracy through hyperparameter tuning.
- Deploy the application online.
- Add visualizations and more property features.

## 👩‍💻 Author

**Pavithra Daddali**

GitHub: [Pavithra-web68](https://github.com/Pavithra-web68)

---

⭐ If you find this project useful, consider giving the repository a star.