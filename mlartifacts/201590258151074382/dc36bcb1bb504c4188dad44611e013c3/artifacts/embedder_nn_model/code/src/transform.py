from services.airflow.dags.data_prepare import data_prepare_pipeline

def main():
    print("Testing zenml pipeline")
    data_prepare_pipeline()
    print("Zenml pipeline successfully completed!")


if __name__ == "__main__":
    main()