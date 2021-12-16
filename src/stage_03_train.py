import pandas as pd

import os.path
import argparse
import joblib

from sklearn.linear_model import ElasticNet
from src.utils.all_utils import read_yaml,create_directory, save_local_df


def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # save dataset in the local directory
    # create path for directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']

    train_data_filename = config['artifacts']['train']

    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)

    train_data = pd.read_csv(train_data_path)

    train_y = train_data["quality"]
    train_x = train_data.drop(columns="quality", axis=1)

    alpha_ = params['model_params']['ElasticNet']['alpha']
    l1_ratio_ = params['model_params']['ElasticNet']['l1_ratio']
    random_state_ = params['base']['random_state']

    lr = ElasticNet(alpha=alpha_, l1_ratio=l1_ratio_, random_state=random_state_)
    lr.fit(train_x,  train_y)

    model_dir = config['artifacts']['model_dir']
    model_filename = config['artifacts']['model_file']

    model_dir = os.path.join(artifacts_dir, model_dir)
    create_directory([model_dir])
    model_path = os.path.join(model_dir, model_filename)

    joblib.dump(lr, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    train(config_path=parsed_args.config, params_path=parsed_args.params)
