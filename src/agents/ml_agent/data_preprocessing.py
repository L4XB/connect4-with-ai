import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def preprocess_data():
    # load dataset
    connect_4 = fetch_ucirepo(id=26)
    
    # extract features and labels
    X = connect_4.data.features
    y = connect_4.data.targets
    
    # koordinate board state
    X_encoded = X.applymap(lambda x: 1 if x == 'x' else (-1 if x == 'o' else 0))
    
    # korrdinate labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.squeeze())
    
    # create train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded.values,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test