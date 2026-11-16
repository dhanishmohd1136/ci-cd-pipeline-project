import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.train import train_model

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT/"data"/"data.csv"

MODEL_NAME = "CustomerChurn"

mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
mlflow.set_experiment("ChurnExperiment")

def get_dataset_version(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():

    dataset_version = get_dataset_version(DATA_PATH)

    with mlflow.start_run():

        model,X_train,metrics = train_model(DATA_PATH)

        mlflow.log_params({
            "model":"Logistic Regression",
            "random_state":"43"
        })

        mlflow.log_metrics(metrics)


        modle_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format='cloudpickle',
            input_example=X_train.head(3)
        )

        registerd = mlflow.register_model(
            model_uri=modle_info.model_uri,
            name= MODEL_NAME
        )

        if metrics['f1'] >0.80:
            client = MlflowClient()

            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias='champion',
                version=registerd.version
            )

            print("\n model is promoted to champion")
        else:
            print("\n Model not promoted")

        print("\n Metrics\n",metrics)
        print("\nDataset version :",dataset_version)
        print("\nModel version : ",registerd.version)



if __name__ == "__main__":
    main()