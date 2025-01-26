import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data():
    # load data
    data = pd.read_csv('connect-4.data', header=None)
    
    # split features and labels
    X = data.iloc[:, :42]
    y = data.iloc[:, 42]

    # koordinate featzre
    X_encoded = X.applymap(lambda x: 1 if x == 'x' else (-1 if x == 'o' else 0))
    
    # koordinate labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # create train and test slit
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded.values, 
        y_encoded, 
        test_size=0.2, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test