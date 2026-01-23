
# 🏠 Housing Price Prediction – End-to-End Machine Learning Project

An end-to-end **Housing Price Prediction** system built using **Python, Scikit-Learn, Flask, Docker, and CI/CD**.
This project demonstrates **production-ready ML practices**, from data preprocessing and model training to API deployment and automated testing.

---

## 🚀 Project Overview

Accurate housing price estimation is critical for real-estate platforms, investors, and urban planners.
This project builds a **machine learning regression pipeline** to predict house prices based on structured features such as location, income, and housing characteristics.

### Key Highlights

* End-to-end ML lifecycle implementation
* Modular and production-ready code structure
* REST API for real-time predictions
* Dockerized application
* CI pipeline with automated testing

---

## 🧠 Machine Learning Workflow

1. **Data Ingestion**

   * Load housing dataset (`housing.csv`)
2. **Data Preprocessing**

   * Handle missing values
   * Feature scaling & transformation
3. **Model Training**

   * Regression model training
   * Model evaluation & persistence
4. **API Development**

   * Flask-based inference API
5. **Testing & CI**

   * Unit tests with GitHub Actions
6. **Containerization**

   * Dockerized deployment

---

## 📂 Project Structure

```
housing-price-prediction-main/
│
├── app.py                 # Flask API for predictions
├── train.py               # Model training pipeline
├── preprocess.py          # Data preprocessing logic
├── requirements.txt       # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore
├── .gitignore
│
├── data/
│   └── housing.csv        # Dataset
│
├── tests/
│   └── test_model.py      # Unit tests
│
├── .github/
│   └── workflows/
│       └── ci.yml         # CI pipeline (GitHub Actions)
│
└── README.md
```

---

## 📊 Dataset Description

The dataset contains structured housing information such as:

* Median income
* Housing age
* Number of rooms
* Population
* Geographical features

Target Variable:

* **Median House Value**

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/housing-price-prediction.git
cd housing-price-prediction-main
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

Train the model using:

```bash
python train.py
```

This will:

* Preprocess the data
* Train the regression model
* Save the trained model for inference

---

## 🌐 Run the Prediction API

Start the Flask application:

```bash
python app.py
```

API will be available at:

```
http://localhost:5000
```

### Example Prediction Request

```json
POST /predict
{
  "median_income": 4.5,
  "housing_median_age": 25,
  "total_rooms": 1800,
  "population": 850
}
```

---

## 🧪 Run Tests

```bash
pytest
```

Tests validate:

* Model training
* Prediction consistency

---

## 🐳 Docker Support

### Build Docker Image

```bash
docker build -t housing-price-prediction .
```

### Run Container

```bash
docker run -p 5000:5000 housing-price-prediction
```

---

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** to:

* Run tests automatically on every push
* Ensure code quality and reliability

Workflow file:

```
.github/workflows/ci.yml
```

---

## 📈 Business Impact

* Enables **data-driven pricing decisions**
* Scalable API for integration with real-estate platforms
* Demonstrates production-level ML engineering skills

---

## 🛠 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-Learn**
* **Flask**
* **Docker**
* **GitHub Actions**
* **Pytest**

---

## 👤 Author

**Agastiabhi**
Master’s Student | Data Engineering & Applied Machine Learning

---

## ⭐ Future Enhancements

* Add advanced models (XGBoost, LightGBM)
* Hyperparameter tuning
* Cloud deployment (AWS / GCP)
* Frontend dashboard for predictions

---


