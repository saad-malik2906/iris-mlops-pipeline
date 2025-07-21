# 🤖 End-to-End MLOps Pipeline: Train, Track, and Deploy ML Models with CI/CD

---

## 📋 Project Summary

This project demonstrates the development and deployment of a **complete MLOps pipeline** for a machine learning model using modern tools and automation practices. It showcases how to go from raw data to a live API serving real-time predictions — all integrated with CI/CD.

The goal is to mirror real-world workflows for managing machine learning in production: **automated training**, **model versioning**, **artifact logging**, and **fast, safe deployment** using tools like **MLflow**, **Docker**, **FastAPI**, and **GitHub Actions**.

---

## 🔧 What We Built

- A complete **machine learning pipeline** that:
  - Loads and preprocesses the Iris dataset.
  - Trains a classifier
  - Evaluates and logs model performance.
  - Tracks experiments with **MLflow**.
- A **FastAPI** backend that loads and serves the trained model through a prediction endpoint.
- A **Dockerized application** for consistent, reproducible deployments.
- A **CI/CD pipeline** using **GitHub Actions** that:
  - Trains the model
  - Registers the model with MLflow
  - Builds and pushes a Docker image to Docker Hub
  - Triggers deployment to **Railway**
- **Secrets management** via GitHub and Railway for secure credentials and environment variables.

---

## 🔑 Key Concepts Covered

### Machine Learning

- **Data Splitting & Training:** Using `scikit-learn` to train models from structured data.
- **Model Evaluation:** Calculating metrics like accuracy to judge model quality.
- **Model Logging:** Using **MLflow** to track hyperparameters, metrics, and artifacts for every run.
- **Model Registry:** Saving trained models in a version-controlled registry.

### CI/CD for ML (MLOps)

- **Training Pipeline:** Each push can trigger retraining with updated code/data.
- **Automated Logging:** Automatically logs model details to MLflow.
- **Docker Containerization:** Packages the serving app with the trained model into a portable container.
- **GitHub Actions Workflow:** Automates testing, building, image publishing, and deployment.
- **Deployment on Railway:** The live API gets updated automatically with each successful workflow run.

---

## 🛠️ Tools and Technologies

| Tool/Technology     | Role in Project                                               |
|--------------------|----------------------------------------------------------------|
| **Python**          | Core programming language for ML and web API                  |
| **scikit-learn**    | ML library for model training and evaluation                  |
| **MLflow**          | Tracks experiments, logs models, stores artifacts             |
| **FastAPI**         | Web framework for building the model-serving API              |
| **Docker**          | Containerizes the application and its dependencies            |
| **Docker Hub**      | Stores and shares Docker images publicly or privately         |
| **GitHub Actions**  | Runs CI/CD workflows for automation                           |
| **Railway**         | Cloud platform for deploying the API                          |
| **YAML**            | Workflow definition for GitHub Actions                        |
| **.env files**      | Securely manage environment variables                         |

---

## 🔄 Workflow in Action

1. **Push code to GitHub** (typically to the `main` branch).
2. **GitHub Actions Workflow Triggers Automatically:**
   - Checks out code and sets up Python.
   - Installs dependencies and trains the model.
   - Logs metrics, plots, and the trained model to MLflow.
   - Builds a Docker image containing the FastAPI app and model.
   - Pushes the image to Docker Hub.
   - Triggers Railway deployment.
3. **Railway pulls the latest image and deploys it to production.**
4. **Users can send HTTP requests to the FastAPI endpoint** and receive model predictions in real-time.

---

## 🧪 Example API Usage

After deployment, send a POST request to the `/predict` endpoint like this:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

And you will receive a response such as:

```json
{
  "prediction": "Iris-setosa"
}
```

---

## 📁 Project Structure

```
├── mlops_pipeline.py         # Main training and MLflow logging script
├── app.py                    # FastAPI app to serve the trained model
├── Dockerfile                # Docker instructions for building the app
├── requirements.txt          # Python dependencies
├── .github/
│   └── workflows/
│       └── ci-cd.yml         # GitHub Actions workflow file
├── model/                    # Contains trained model and evaluation plots
├── .gitignore                # Prevents sensitive files from being committed
└── README.md                 # Project documentation
```

---

## 🧠 Lessons & Takeaways

- **MLOps bridges ML and DevOps** — ensuring your models are reproducible, testable, and deployable.
- **MLflow brings transparency** to your experiments by tracking every run.
- **CI/CD for ML pipelines** reduces human error and improves delivery speed.
- **Containerization** ensures consistency between dev, test, and prod environments.
- **Automated cloud deployment** (e.g., with Railway) simplifies operations and enables rapid iteration.

---

## 🌐 Real-World Relevance

This project mirrors what real MLOps teams do daily at companies deploying machine learning in production. Skills and practices from this project can scale to:

- Multi-model systems
- Model retraining pipelines
- Model monitoring and alerts
- Canary/blue-green deployments
- Integration with Kubernetes or cloud platforms like AWS/GCP

---

## 🚀 Future Improvements

- Add model drift detection and alerts.
- Implement model monitoring (e.g., Prometheus + Grafana).
- Set up retraining triggers on new data.
- Add security scanning to the CI/CD pipeline.
- Deploy with HTTPS and authentication.

---

## ✅ Status

✅ Completed — End-to-end MLOps pipeline is implemented and live via Railway.

---
