import numpy as np

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sklearn


from prodml.data import load_data , add_duration , clean_trips , split_trips
from prodml.features import add_route_feature, df_to_dict, MODEL_FEATURES

from prodml.config import settings, Settings

from ast import Dict
from typing import Any 



from prodml.logging_conf import setup_logging , get_logger

logger  = get_logger(__name__)

def train_model(config: Settings = settings )-> Dict[str, float]:
    """Train and persist the ride-duration model."""

    df = load_data(config.data_path)

    df = add_duration(df)

    df = clean_trips(df, min_duration= config.min_duration, max_duration= config.max_duration)
    df = add_route_feature(df= df )

    df_train , df_val = split_trips(df , test_size= config.test_size, random_state= config.random_state)

    train_dict = df_to_dict(df= df_train)
    val_dict = df_to_dict(df= df_val)

    vectorizer = DictVectorizer()
    X_train =  vectorizer.fit_transform(train_dict)
    X_val = vectorizer.fit(val_dict)

    y_train = df_train["duration"].to_numpy()
    y_val = df_val["duration"].to_numpy()


    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(
             y_val,
             y_pred,
          )


    mse = mean_squared_error(
        y_val,
        y_pred,
    )

    rmse = np.sqrt(mse)

    artifact: dict[str, Any] = {
        "vectorizer": vectorizer,
        "model": model,
        "metadata": {
            "model_version": config.model_version,
            "feature_names": MODEL_FEATURES,
            "framework": (
                f"scikit-learn {sklearn.__version__}"
            ),
            "metrics": {
                "mae": mae,
                "rmse": rmse,
            },
        },
    }

    with config.model_path.open("wb") as f:
        pickle.dump(artifact, f)



    logger.info("model.trained",
        extra={
            "mae": mae,
            "rmse": rmse,
            "training_rows": len(df_train),
            "validation_rows": len(df_val),
            "model_path": str( config.model_path ) 
            }
            )



    return {
            "mae": mae,
            "rmse": rmse,
        }


def main()-> None:
    setup_logging(settings.log_level)
    train_model(settings)


# command-line entry point
if __name__ == "__main__":
    main()