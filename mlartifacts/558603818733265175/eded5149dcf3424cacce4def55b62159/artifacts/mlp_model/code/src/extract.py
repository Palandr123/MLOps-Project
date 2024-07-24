from subprocess import run

def main():
    print("Testing extract_data dag")
    run(
        [
            "airflow",
            "dags",
            "test",
            "extract_data"
        ],
        check=True,
    )
    print("Dag completed successfully")

if __name__ == "__main__":
    main()