import random
import json
import requests
import hydra
from model import load_features
import zenml

@hydra.main(config_path="../configs", config_name="main", version_base=None) # type: ignore
def predict(cfg = None):    

    X, y = load_features(name = "features_target", 
                        version = cfg.data_version, 
                        )

    idx = random.randint(0, y.shape[0] - 1)
    example = X.iloc[idx,:]
    example_target = y[idx]   

    example = json.dumps(   
        {"inputs" : example.to_dict()}
    )
    print(example)

    response = requests.post(
        url=f"http://localhost:{cfg.port}/invocations",
        data=example,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("actual price: ", example_target)


if __name__=="__main__":
    predict()