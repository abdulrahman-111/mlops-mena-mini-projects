from typing import Any
from unittest import result
from ast import Dict

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


from prodml.logging_conf import get_logger, timed 

MODEL_FEATURES = ["PU_DO","trip_distance"]



# for training and enigineer feature 
def add_route_feature(df: pd.DataFrame)-> pd.DataFrame:
    """Create pickup-dropoff route category."""

    result = df.copy()

    result["PU_DO"] =  ( result["PULocationID"].astype(str)
                      + "_" +  
                     result["DOLocationID"].astype(str)
                    )

    return result



# for training 
def df_to_dict(df:pd.DataFrame)-> list[ Dict[ str, Any ] ]:
    """ Transform from df to list of dictionaries """


    features: list[str] = MODEL_FEATURES
    dicts = df[features].to_dict(orient="records") #Mlist of dicts

    return dicts

# for API 
def make_prediction_features(pickup_id: int, dropoff_id:int , trip_distance:float) -> Dict[str, Any]:
    """Build model input for one prediction."""
    input_pu_do = str(pickup_id) + "_" + str(dropoff_id)

    return {
        "PU_DO": input_pu_do,
        "trip_distance": trip_distance
    }