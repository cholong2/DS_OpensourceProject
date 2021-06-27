import pandas as pd
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor



def bestSearch (param, df, target):
    '''
    description : A function that finds the best combination of scale and model with only numeric columns

    :param param: Dictionary data type, 'scaler' and 'model' are key values.
    :param df: Data to scale
    :param target: Column to predict
    :return: Returns the best combination with the highest score.
    '''

    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))
    bestDi = {}

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=33)
    for s in scaler:
        X_train_scale, X_test_scale = scaled(X_train, X_test, s)
        for m in model:
            bestDi[ s + ", " + m] = predict(m, X_train_scale, X_test_scale, y_train, y_test)
            print( s + ", " + m, bestDi[ s + ", " + m])

    return max(bestDi, key=bestDi.get), max(bestDi.values())



def bestSearchEncoding(param, df, target, encoding_cols, scaling_cols):
    '''
    description : A function that finds the optimal combination of scalers, models, and encoders in data containing categorical variables

    :param param:  Dictionary data type, 'scaler', 'model', 'encoding' are key values.
    :param df: Data to scale and encode
    :param target: Column to predict
    :param encoding_cols: Column to encode
    :param scaling_cols: Column to scale
    :return: Returns the best combination with the highest score.
    '''

    encoder = np.array(param.get('encoder'))
    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))
    bestDi = {}

    for e in encoder:
        encode_df = encoding(e, encoding_cols, df)
        X_train, X_test, y_train, y_test = train_test_split(encode_df, target, test_size=0.2, random_state=33)
        for s in scaler:
            X_train_scale, X_test_scale = scaled(X_train, X_test, s)
            for m in model:
                bestDi[e+", "+s+", "+m] = predict(m,  X_train_scale, X_test_scale, y_train, y_test)

    return max(bestDi, key=bestDi.get), max(bestDi.values())


def scaled (X_train, X_test, scaler):
    '''
    Description : A function that scales to the scale received as a parameter.

    :param X_train: train data
    :param X_test: test data
    :param scaler: Scaler to use, scaler has 'standard', 'minmax', and 'robust'.
    :return: scaled train data, test data
    '''
    if (scaler == "standard"):
        stdScaler = StandardScaler()
        X_train_scale = stdScaler.fit_transform(X_train)
        X_test_scale = stdScaler.transform(X_test)
        return X_train_scale, X_test_scale

    elif (scaler == "robust"):
        robustScaler = RobustScaler()
        X_train_scale = robustScaler.fit_transform(X_train)
        X_test_scale = robustScaler.transform(X_test)
        return X_train_scale, X_test_scale

    elif (scaler == "minmax"):
        mmScaler = MinMaxScaler()
        X_train_scale = mmScaler.fit_transform(X_train)
        X_test_scale = mmScaler.transform(X_test)
        return X_train_scale, X_test_scale



def encoding (encoder,cols, df):
    '''
    Description:  A function that replaces categorical columns with numeric columns

    :param encoder: Encode to use, encoder has 'labelEncoder', 'oneHotEncoder'
    :param cols: Categorical columns
    :param df: data to encode
    :return: encoded data
    '''
    if (encoder=="labelEncoder"):
        label_df = df.copy()
        for c in cols:
         lb = LabelEncoder()
         lb.fit(list(df[c].values))
         label_df[c] = lb.transform(list(df[c].values))

        return label_df

    elif (encoder == "oneHotEncoder"):
        onehot_df = df.copy()
        for c in cols:
            onehot_df = pd.get_dummies(data=onehot_df, columns=[c])

        return onehot_df



def predict (model, X_train_scale, X_test_scale, y_train, y_test):
    '''
    Description: A function that learns targets using models received with scale and encoded data, and to predict targets with learned models.

    :param model: Model to use for learning, model has '"adaboost", "decisiontree", "bagging", "XGBoost", "randomforest" and "gradient"
    :param X_train_scale: Scale and encoded data for learning
    :param X_test_scale: Data to use for predictions
    :param y_train: Target data for learning
    :param y_test: Target data to use for predictions
    :return: Returns the score of the model.
    '''

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    if (model == "adaboost"):
        #AdaBoostRegressor
        ada_reg = AdaBoostRegressor()
        ada_param = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1]
        }
        ada = GridSearchCV(ada_reg, param_grid=ada_param, cv=kfold)
        ada.fit(X_train_scale, y_train)
        return ada.score(X_test_scale, y_test)

    elif(model == "decisiontree"):
        #DecisionTreeRegressor
        decision_tree_model = DecisionTreeRegressor()
        param_grid = {
            'criterion': ['mse'],
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=param_grid, cv=kfold)
        gsDT.fit(X_train_scale, y_train)
        return  gsDT.score(X_test_scale, y_test)

    elif (model == "bagging"):
        #BaggingRegressor
        bagging = BaggingRegressor()
        b_param_grid = {
            'n_estimators': [10, 50, 100],
            'max_samples': [1, 5, 10],
            'max_features': [1, 5, 10]
        }
        gsBagging = GridSearchCV(bagging, param_grid=b_param_grid, cv=kfold)
        gsBagging.fit(X_train_scale, y_train)
        return gsBagging.score(X_test_scale, y_test)

    elif (model == "XGBoost"):
        #XGBRegressor
        XGB = XGBRegressor()
        xgb_param_grid = {
            'learning_rate': [1, 0.1, 0.01],
            'max_depth': [1, 5, 10, 50]
        }
        gsXGB = GridSearchCV(XGB, param_grid=xgb_param_grid, cv=kfold)
        gsXGB.fit(X_train_scale, y_train)
        return gsXGB.score(X_test_scale, y_test)

    elif (model == "randomforest"):
        #RandomForestRegressor
        forest = RandomForestRegressor()
        fo_grid = {
            "n_estimators": [50, 100, 200, 500],
            "criterion":["mse", "mae"],
            "max_depth": [None, 2, 3, 4, 5, 6]
        }
        gsRd = GridSearchCV(forest, param_grid=fo_grid, cv=kfold)
        gsRd.fit(X_train_scale, y_train)
        return gsRd.score(X_test_scale, y_test)

    elif (model == "gradient"):
        #GradientBoostingRegressor
        gbr = GradientBoostingRegressor()
        param = {
            "n_estimators": [25, 50, 100],
            "max_depth": [1, 2, 4],
            "learning_rate": [1, 0.1, 0.01],
            "subsample": [1, 0.5, 0.01]
        }
        gsGd = GridSearchCV(gbr, param_grid=param, cv=kfold)
        gsGd.fit(X_train_scale, y_train)
        return gsGd.score(X_test_scale, y_test)

