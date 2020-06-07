
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneGroupOut
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler

from ..Config import Config


class Impute(Config):

    # put the global variables in the config
    # no action
    # filter only the available data for training

    def __init__(self, impute_method=None):
        self.impute_method = impute_method or self.MODELLING_CONFIG["IMPUTE_METHOD"]


    def impute(self, data):
        """impute missing data from the selection of imputer methods"""
        imputer = {
            "knn": self.knn(data, n_neighbors=5, weights="uniform", metric="nan_euclidean"),
            "interpolate_ffill": self.interpolate(data, method="linear", limit_direction="forward"),
            "interpolate_bfill": self.interpolate(data, method="linear", limit_direction="both"),
            "simple": self.simple(data, missing_values=np.nan, strategy='mean'),
            "iterative": self.iterative(data, missing_values=np.nan, random_state=Config.MODELLING_CONFIG["RANDOM_STATE"], n_nearest_features=5)
        }
        
        return imputer.get(self.impute_method, "Invalid impute method")


    def knn(self, data, n_neighbors=5, weights="uniform", metric="nan_euclidean"):
        
        # # setting up KNN imputer
        knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)

        # # prep data for imputation
        temp_data = data.dropna(axis='columns', how = 'all')
        scaler = MinMaxScaler()
        temp_vars = [col for col in temp_data.columns if col in self.vars()]  
        scaler.fit(temp_data[temp_vars])
        temp_data_scaled = scaler.transform(temp_data[temp_vars])

        # # imputing na with KNN imputer 
        array_scaled = knn_imputer.fit_transform(temp_data_scaled)
        array = scaler.inverse_transform(array_scaled)

        df = pd.DataFrame(array)
        index = data.index
        df.index = index
        temp_data = data.copy()
        temp_data.index = index
        temp_data[temp_vars] = df

        return temp_data
        

    def interpolate(self, data, method="linear", limit_direction=None):
        limit_direction = limit_direction or self.EDA_CONFIG["INTERPOLATE_DIRECTION"]
        
        data = data.interpolate(method=method, limit_direction=limit_direction)

        return data

