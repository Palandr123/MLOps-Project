from zenml.client import Client

def load_features(name, version, size = 1):
    client = Client()

    # Fetch all artifacts with the specified name and version
    artifacts = client.list_artifacts(name=name, version=version)
    
    # Sort artifacts by version if needed
    artifacts = sorted(artifacts, key=lambda x: x.version, reverse=True)

    df = artifacts[0].load()
    df = df.sample(frac = size, random_state = 88)

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    X = df[df.columns[:-1]]
    y = df[[df.columns[-1]]]

    print("shapes of X,y = ", X.shape, y.shape)

    return X, y
