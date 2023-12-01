import logging
import json
import os
import glob
import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    logger.debug('Hello my custom SageMaker init script!')
    f_output_model = open("/opt/ml/model/my-model.txt", "w")
    f_output_model.write(json.dumps(glob.glob("{}/*/*/*.*".format(os.environ['SM_INPUT_DIR']))))
    f_output_model.close()
    
    f_output_data = open("/opt/ml/output/data/my-data.txt", "w")
    f_output_data.write(json.dumps(dict(os.environ), sort_keys=True, indent=4))
    f_output_data.close()

    data = pd.read_csv('{}/data/training/my-input-csv-file.csv'.format(os.environ['SM_INPUT_DIR']), low_memory=False)
    
    # metto a -1 gli score mancanti
    data = data.replace(np.nan, -1)

    y = data['LABEL']
    X = data.drop(columns=['LABEL'])
    # X = data[feature]


    X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.20, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=123, stratify=y_train)

    automl = AutoML()
    settings = {
        "time_budget": -1,  # total running time in seconds
        "metric": 'macro_f1', 
        "task": 'classification',  # task type
        "log_file_name": 'result/variant_training.log',
        "estimator_list": ['xgboost'],
        "max_iter": 100,
    }

    automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings)


    y_pred = automl.predict(X_test)

