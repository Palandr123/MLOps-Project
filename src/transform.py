from services.airflow.dags.data_prepare import extract, transform, validate, load


def main():
    print("Testing zenml pipeline")
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    df = load(X, y, version)
    print("Zenml pipeline successfully completed!")


if __name__ == "__main__":
    main()