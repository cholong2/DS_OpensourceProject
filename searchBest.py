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
    encoder = np.array(param.get('encoding'))
    scaler = np.array(param.get('scaler'))
    model = np.array(param.get('model'))
    bestDi = {}

    for e in encoder:
        encode_df = encoding(e, encoding_cols, df)
        X_train, X_test, y_train, y_test = train_test_split(encode_df[scaling_cols], target, test_size=0.2, random_state=33)
        for s in scaler:
            X_train_scale, X_test_scale = scaled(X_train, X_test, s)
            for m in model:
                bestDi[e+", "+s+", "+m] = predict(m,  X_train_scale, X_test_scale, y_train, y_test)

    return max(bestDi, key=bestDi.get), max(bestDi.values())


def scaled (X_train, X_test, scaler):
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
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    if (model == "adaboost"):
        # adaboost
        ada_reg = AdaBoostRegressor()
        ada_param = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1]
        }
        ada = GridSearchCV(ada_reg, param_grid=ada_param, cv=kfold)
        ada.fit(X_train_scale, y_train)
        return ada.score(X_test_scale, y_test)

    elif(model == "decisiontree"):
        #decisiontreeRegressor
        decision_tree_model = DecisionTreeRegressor()
        param_grid = {
            'criterion': ['mse'],
            'max_depth': [None, 2, 3, 4, 5, 6]
        }
        gsDT = GridSearchCV(decision_tree_model, param_grid=param_grid, cv=kfold)
        gsDT.fit(X_train_scale, y_train)
        return  gsDT.score(X_test_scale, y_test)

    elif (model == "bagging"):
        # bagging
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
        # XGBoost
        XGB = XGBRegressor()
        xgb_param_grid = {
            'learning_rate': [1, 0.1, 0.01],
            'max_depth': [1, 5, 10, 50]
        }
        gsXGB = GridSearchCV(XGB, param_grid=xgb_param_grid, cv=kfold)
        gsXGB.fit(X_train_scale, y_train)
        return gsXGB.score(X_test_scale, y_test)

    elif (model == "randomforest"):
        #RandomForest
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
        # gradient boosting
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

