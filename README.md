# 🌬️ Air Quality Prediction MLOPs Project

An end-to-end MLOps pipeline for predicting air quality using machine learning. This project demonstrates automated data collection, model training, and model deployment with CI/CD practices.

## 🚀 Features

- **Automated Data Pipeline**: Fetches air quality data daily from the AQICN API
- **ML Model Training**: Trains and evaluates a Random Forest Regressor to predict AQI
- **Model Registry**: Tracks experiments and manages model versions using MLflow
- **CI/CD Pipeline**: Automated testing, training, and deployment with GitHub Actions
- **Monitoring**: Tracks model performance and data quality over time

## 📦 Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) (for dependency management)
- MLflow server (for experiment tracking)
- AQICN API token (get one from [aqicn.org](https://aqicn.org/data-platform/token/))

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/air-quality-mlops.git
   cd air-quality-mlops
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your AQICN token and other settings
   ```

## 🚦 Usage

### Data Pipeline

- **Manual Run**:
  ```bash
  python fetch_data_github.py --start-date 2025-01-01 --end-date 2025-01-07
  ```

- **Scheduled Run**:
  The data pipeline runs automatically daily via GitHub Actions

### Model Training

- **Local Training**:
  ```bash
  python train_model.py --test-size 0.2 --n-estimators 100
  ```

- **Scheduled Training**:
  The model is automatically trained every Monday via GitHub Actions

### Model Serving

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

## 🔄 Workflows

### 1. Data Pipeline (`fetch_data.yml`)
- Fetches air quality data daily
- Validates and processes new data
- Commits updates to the repository

### 2. Model Training (`train_model.yml`)
- Trains model weekly with the latest data
- Evaluates model performance
- Updates model registry if performance improves
- Generates model cards and visualizations

## 📊 Project Structure

```
air-quality-mlops/
├── .github/workflows/    # GitHub Actions workflows
├── data/                 # Raw and processed data
├── models/               # Trained models and artifacts
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Reports and visualizations
├── src/                  # Source code
│   ├── __init__.py
│   ├── data/            # Data processing modules
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and evaluation
│   └── monitoring/      # Model and data monitoring
├── tests/               # Unit and integration tests
├── .env.example         # Environment variables template
├── poetry.lock          # Dependency lock file
├── pyproject.toml       # Project metadata and dependencies
└── README.md            # This file
```

## 📈 Model Performance

Current model metrics:
- **R² Score**: 0.85
- **MAE**: 12.3
- **RMSE**: 15.6

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AQICN](https://aqicn.org/) for the air quality data API
- [MLflow](https://mlflow.org/) for experiment tracking
- [Scikit-learn](https://scikit-learn.org/) for machine learning
- [FastAPI](https://fastapi.tiangolo.com/) for the API server

---

<div align="center">
  Made with ❤️ by Your Name
</div>
