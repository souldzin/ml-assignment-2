import textwrap
import pandas as pd
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone

# --------------------------------------------
# constants
# --------------------------------------------

DEFAULT_TRAIN_DATA_PATH     = "./adultTrain.data"
DEFAULT_TEST_DATA_PATH     = "./adultTest.data"
DATA_HEADERS        = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
    "occupation", "relationship", "race", "sec", "capital-gain", "capital-loss", 
    "hours-per-week", "native-country", "y"
]
DUMMY_COLUMNS       = [
    "workclass", 
    "education", 
    "marital-status", 
    "occupation", 
    "relationship", 
    "race", 
    "sec",
    "native-country"
]
CLEAN_EDUCATION_MAP = {
    "HS-grad":      "High-school",
    "Some-college": "Higher-education",
    "Bachelors":    "Undergraduate",
    "Masters":      "Graduate",
    "Assoc-voc":    "Higher-education",
    "11th":         "Grade-school",
    "Assoc-acdm":   "Higher-education",
    "10th":         "Grade-school",
    "7th-8th":      "Grade-school",
    "Prof-school":  "Higher-education",
    "9th":          "Grade-school",              
    "12th":         "Grade-school",
    "Doctorate":    "Graduate",
    "5th-6th":      "Grade-school",
    "1st-4th":      "Grade-school",
    "Preschool":    "Grade-school"
}
CLEAN_WORKCLASS_MAP = {
    "Private":          "Private",
    "Self-emp-not-inc": "Self-employed",
    "Self-emp-inc":     "Self-employed",
    "Local-gov":        "Government",
    "State-gov":        "Government",
    "Federal-gov":      "Government",
    "Without-pay":      "Without-pay"
}

# --------------------------------------------
# data reading, cleaning and parsing functions
# --------------------------------------------

def clean_y(value):
    """Clean a y column value

    Rationale:
    - The adultTest.data has a '.' at the end of the value...
    """
    return value.replace(".", "")

def clean_marital_status(value):
    lowerValue = value.lower()
    
    if lowerValue == "never-married":
        return "Never-married"
    elif "married" in lowerValue:
        return "Married"
    else:
        return "Previously-married"
    
def clean_native_country(value):
    lowerValue = value.lower()
    
    if lowerValue == "united-states":
        return "United-States"
    else:
        return "Other"

def clean_data(data):
    # Strip whitespaces from all string values
    # and replace "?" with None,
    # and drop all na rows
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) \
               .replace(["?"], [None]) \
               .dropna()
    # Clean "marital-status"
    data["marital-status"] = data["marital-status"].map(clean_marital_status)
    # Clean "native-country"
    data = data[data["native-country"] != "?"]
    data["native-country"] = data["native-country"].map(clean_native_country)
    # Clean "education"
    data["education"] = data["education"].map(CLEAN_EDUCATION_MAP)
    # Clean "workclass"
    data["workclass"] = data["workclass"].map(CLEAN_WORKCLASS_MAP)
    # Clean "y"
    data["y"] = data["y"].map(clean_y)
    # Drop unecessary columns
    # - education-num - this looks like an identifier for the original education value (not needed!)
    data.drop(["fnlwgt", "capital-gain", "capital-loss", "education-num"], axis=1, inplace=True)
    return data

def prepare_data(data):
    return pd.get_dummies(data, columns=DUMMY_COLUMNS)

def read_data(path):
    dataset = pd.read_csv(path)
    dataset.columns = DATA_HEADERS
    dataset = clean_data(dataset)
    dataset = prepare_data(dataset)
    return dataset

def get_data_xy(data):
    X = data.loc[:, data.columns != "y"]
    y = data.loc[:, "y"]

    return X, y

# ------------------------------------
# model training and testing functions
# ------------------------------------

def test_model_kfold(model, X, y):
    """Run a kfold test on the given model. It works off of a clone of the given model."""
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None, shuffle=True)
    avg_score = 0
    avg_cnf_matrix = None
    
    # For each fold...
    # 1) clone the model to get a fresh copy
    # 2) train and test the model on the split
    # 3) aggregate the test results
    for train_index, test_index in kf.split(X):       
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        
        model = clone(model)
        model.fit(train_X, train_y)
        score, cnf_matrix = test_model(model, test_X, test_y)
        
        avg_score += score
        avg_cnf_matrix = avg_cnf_matrix + np.matrix(cnf_matrix) if avg_cnf_matrix is not None else np.matrix(cnf_matrix)
        
    avg_score = avg_score / num_folds
    avg_cnf_matrix = avg_cnf_matrix / num_folds
    
    return (avg_score, avg_cnf_matrix)

def test_model_split(model, train_split_xy):
    """Run a test on the given split. It works off of a clone of the given model."""
    train_X, test_X, train_y, test_y = train_split_xy
    model = clone(model)
    model.fit(train_X, train_y)
    return test_model(model, test_X, test_y)
    
def test_model (model, test_X, test_y):
    """Get performance metrics based on the model's prediction results."""
    predicted = model.predict(test_X)
    cnf_matrix = confusion_matrix(test_y, predicted)
    score = accuracy_score(test_y, predicted)
    return (score, cnf_matrix)

def print_test_results (results):
    score, cnf_matrix = results
    print("Accuracy Score:")
    print(textwrap.indent(str(score), " " * 4))
    print("")
    print("Confusion Matrix:")
    print(textwrap.indent(str(cnf_matrix), " " * 4))

def prepare_models(models, X, y):
    """Prepare the given models and print training results"""
    xy_split = train_test_split(X, y, test_size=0.25)
    for model_name, model in models:
        print("'{0}' model".format(model_name))
        print("--------------------------------------")

        print("Testing against training data...")
        print("")

        test_results = test_model_split(model, xy_split)
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
            ("Neural Network", MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam")),
            ("SVM", SVC(C=1.0, kernel="rbf")),
            ("KNN", KNeighborsClassifier(n_neighbors=10))
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