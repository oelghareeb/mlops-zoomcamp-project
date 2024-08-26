# src/train.py
import os
import mlflow
from model import build_model  # Adjust the import path if necessary
from data_preprocessing import load_and_preprocess_data  # Adjust the import path if necessary
from mlflow.models.signature import infer_signature  # Model Signature

def main():
    # Set MLflow experiment
    mlflow.set_experiment("FashionMNIST_Experiment")

    with mlflow.start_run():
        # Load and preprocess data
        train_filepath = "./data/fashion-mnist_train.csv"
        test_filepath = "./data/fashion-mnist_test.csv"
        X_train, y_train, X_test, y_test = load_and_preprocess_data(train_filepath, test_filepath)

        # Build model
        model = build_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        # Log metrics to MLflow
        for metric_name, metric_value in history.history.items():
            mlflow.log_metric(metric_name, metric_value[-1])  # Log the last value of each metric

        # Log model to MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        artifact_path = os.path.join(os.getcwd(), 'artifact_dir')
        mlflow.keras.log_model(model, artifact_path, signature = signature)

if __name__ == "__main__":
    main()
