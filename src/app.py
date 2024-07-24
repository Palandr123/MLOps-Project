import gradio as gr
import mlflow
from model import load_features
from data import preprocess_data
import json 
import requests
import numpy as np
import pandas as pd
from hydra import initialize, compose
from get_uniques import get_unique_values



initialize(config_path="../configs", version_base=None)
cfg = compose(config_name='main')

uniques = get_unique_values()

def predict(region = None,
            year = None,
            manufacturer = None,
            model = None,
            #condition = None,
            #cylinders = None,
            fuel = None,
            odometer = None,
            title_status = None,
            #transmission = None,
            VIN = None,
            #drive = None,
            #size = None,
            #type = None,
            #   paint_color = None,
            #county = None,
            state = None,
            lat = None,
            long = None,
            posting_date = None):
    
    # This will be a dict of column values for input data sample
    features = {
        "image_url" : None,
        "description" : None,
        "id" : None,
        "url" : None,
        "region_url" : None,
        "region" : region,
        "year" : year,
        "manufacturer" : manufacturer,
        "model" : model,
        "condition" : None,
        "cylinders" : None,
        "fuel" : fuel,
        "odometer" : odometer,
        "title_status" : title_status,
        "transmission" : None,
        "VIN" : VIN,
        "drive" : None, 
        "size" : None,
        "type" : None,
        "paint_color" : None,
        "county" : None,
        "state" : state,
        "lat" : lat,
        "long" : long,
        "posting_date" : posting_date,
    }
    
    # print(features)
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    # X = transform_data(
    #                     df = raw_df, 
    #                     cfg = cfg, 
    #                     return_df = False, 
    #                     only_transform = True, 
    #                     transformer_version = "v4", 
    #                     only_X = True
    #                   )
    X = preprocess_data(raw_df, drop_rows=False, return_y=False)
    
    # Convert it into JSON
    example = X.iloc[0,:]

    example = json.dumps( 
        { "inputs": example.to_dict() }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    port_number = 5000
    response = requests.post(
        url=f"http://localhost:{port_number}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()

# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    # TODO: add options for dropdowns
    inputs = [
        gr.Textbox(label = 'region'),
        gr.Number(label = 'year'),
        gr.Dropdown(label = 'manufacturer', choices=uniques['manufacturer']),
        gr.Dropdown(label = 'model', choices=uniques['model']),
        #gr.Dropdown(label = 'condition', choices=uniques['condition']),
        #gr.Dropdown(label = 'cylinders', choices=uniques['cylinders']),
        gr.Dropdown(label = 'fuel', choices=uniques['fuel']),
        gr.Number(label = 'odometer'),
        gr.Dropdown(label = 'title_status', choices=uniques['title_status']),
        #gr.Dropdown(label = 'transmission', choices=uniques['transmission']),
        gr.Textbox(label = 'VIN'),
        #gr.Dropdown(label = 'drive', choices=uniques['drive']),
        #gr.Dropdown(label = 'size', choices=uniques['size']),
        #gr.Dropdown(label = 'type', choices=uniques['type']),
        #gr.Dropdown(label = 'paint_color', choices=uniques['paint_color']),
        #gr.Dropdown(label = 'county', choices=uniques['county']),
        gr.Dropdown(label = 'state', choices=uniques['state']),
        gr.Number(label = 'lat'),
        gr.Number(label = 'long'),
        gr.Textbox(label = 'posting_date'),
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="prediction result"),
    
    # This will provide the user with examples to test the API
    examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# Launch the web UI locally on port 5155
demo.launch(server_name='0.0.0.0',  server_port = 5155)