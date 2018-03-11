import pandas as pd
import numpy as np
import textwrap
import sys
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone

# --------------------------------------------
# constants
# --------------------------------------------

FOLD_COUNT          = 5
DEFAULT_TRAIN_DATA_PATH     = "./bikeRentalHourlyTrain.csv"
DEFAULT_TEST_DATA_PATH     = "./bikeRentalHourlyTest.csv"

# --------------------------------------------
# data reading, cleaning and parsing functions
# --------------------------------------------

def get_data_xy(data):
    X = data.loc[:, data.columns != "cnt"]
    y = data.loc[:, "cnt"]
    return X, y

def clean_data(data):
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)            .replace(["?"], [None])            .dropna()
    # Dropping columns...
    # - 0 - the first column contains some sort of identifier
    # - instant - this seems like an identifier too
    # - dteday - for this info, the fields mnth, weekday, holiday, are good enough
    # - season - this can be computed from mnth / weekday
    # - yr - we're not concerned about year over year performance
    # - casual - this is part of "cnt"
    # - registered - this is part of "cnt"
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(['instant', 'dteday', 'season', 'yr', 'casual', 'registered'], axis=1, inplace=True)
    return data

def read_data(path):
    dataset = pd.read_csv(path, header = 0)
    return clean_data(dataset)

# ------------------------------------
# model training and testing functions
# ------------------------------------

def test_model_kfold(model, X, y, fold_count = FOLD_COUNT):
    """Run a kfold test on the given model. It works off of a clone of the given model."""
    kf = KFold(n_splits=fold_count, random_state=None, shuffle=True)
    avg_metrics = None
    
    # For each fold...
    # 1) clone the model to get a fresh copy
    # 2) train and test the model on the split
    # 3) aggregate the test results
    for train_index, test_index in kf.split(X):       
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        
        model = clone(model)
        model.fit(train_X, train_y)
        metrics = test_model(model, test_X, test_y)
        
        if avg_metrics is None:
            avg_metrics = metrics
        else:
            avg_metrics = tuple(map(lambda x, y: x + y, avg_metrics, metrics))
        
    avg_metrics = tuple(map(lambda x: x / fold_count, avg_metrics))
    
    return avg_metrics

def test_model_split(model, train_split_xy):
    """Run a test on the given split. It works off of a clone of the given model."""
    train_X, test_X, train_y, test_y = train_split_xy
    model = clone(model)
    model.fit(train_X, train_y)
    return test_model(model, test_X, test_y)
    
def test_model (model, test_X, test_y):
    """Get performance metrics based on the model's prediction results."""
    prediction = model.predict(test_X)
    
    mse = mean_squared_error(test_y, prediction)
    var_score = r2_score(test_y, prediction)
    y_bar_squared = (sum(test_y)/float(len(test_y)))**2
    mse_per = mse / y_bar_squared
    
    return (mse, mse_per, var_score)

def print_test_results (results):
    mse, mse_per, var_score = results
    print("MSE:")
    print(textwrap.indent(str(mse), " " * 4))
    
    print("")
    print("MSE%:")
    print(textwrap.indent(str(mse_per), " " * 4))
    
    print("")
    print("Variance Score:")
    print(textwrap.indent(str(var_score), " " * 4))

def prepare_models(models, X, y):
    """Prepare the given models and print training results"""
    for model_name, model in models:
        print("'{0}' model".format(model_name))
        print("--------------------------------------")

        print("Testing against training data with {0} folds...".format(FOLD_COUNT))
        print("")

        test_results = test_model_kfold(model, X, y, fold_count=FOLD_COUNT)
        print_test_results(test_results)
        print("")

        print("Training against training data...")
        model.fit(X, y)

        print("--------------------------------------")
        print("")
    return models

def test_models(models, test_X, test_y):
    """Test the given models against the testing data"""
    for model_name, model in models:
        print("Testing '{0}' model against testing data".format(model_name))
        print("--------------------------------------")
        test_results = test_model(model, test_X, test_y)
        print_test_results(test_results)
        print("--------------------------------------")
        print("")

# -------------------------------
# main
# -------------------------------

def main():
    train_data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRAIN_DATA_PATH
    test_data_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEST_DATA_PATH

    print("Reading training data from: {0}".format(train_data_path))
    train_data = read_data(train_data_path)
    train_data_X, train_data_y = get_data_xy(train_data)

    print("Reading testing data from: {0}".format(test_data_path))
    test_data = read_data(test_data_path)
    test_data_X, test_data_y = get_data_xy(test_data)

    print("")
    print("==============")
    print("Prepare models")
    print("==============")
    print("")

    models = prepare_models(
        [
            ("Linear Regression", LinearRegression()),
            ("Neural Network", MLPRegressor(hidden_layer_sizes=(20,100,20), activation='relu', learning_rate='adaptive', max_iter=300)),
            ("KNN", KNeighborsRegressor(n_neighbors=10, p=1))
        ], 
        train_data_X, 
        train_data_y
    )

    print("")
    print("===========")
    print("Test models")
    print("===========")
    print("")

    test_models(models, test_data_X, test_data_y)

if __name__ == "__main__":
    main()