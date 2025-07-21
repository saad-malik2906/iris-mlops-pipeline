# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY app.py ./
COPY best_model.pkl ./

# Install dependencies
RUN pip install fastapi==0.116.1 uvicorn==0.35.0 joblib numpy scikit-learn

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
