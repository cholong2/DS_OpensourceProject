import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import searchBest

pd.set_option('display.max_columns', None)
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
test_df = test
print(train)
print(test_df)

target=train['SalePrice']


# GrLivArea의 outliar 제거
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
#plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#타겟피쳐인 SalePrice와 0.3이상의 상관관계를 갖는 변수들을 출력
cor = train.corr()
cor_fe = cor.index[abs(cor['SalePrice']) >= 0.3]
# 히트맵 사용
plt.figure(figsize=(15,10))
sns.heatmap(train[cor_fe].corr(),annot=True)


#train과 test를 합치기 위한 feature
fe_name = list(test_df)
print(fe_name)

#두개를 df라는 변수에 합쳐줌
df_train = train[fe_name]
df = pd.concat((df_train,test_df)).reset_index(drop=True,inplace=False)
print(train.shape, test_df.shape, df.shape)
print(df)
#널값있는 피쳐 확인
null_df = df.isna().sum()
print("-----------------널값이 있는 피쳐--------------------")
print(null_df[null_df > 0])

#널값이 50%이상인 피쳐 제거(ignore feature)
check_null = df.isna().sum()/len(df)
print("-----------------널값이 50%인 피쳐--------------------")
print(check_null[check_null >= 0.5])
remove_cols = check_null[check_null >= 0.5].keys()
df = df.drop(remove_cols, axis=1)

#결측치 처리
#FireplaceQu: NA는 난로가 없다는것을 의미해서 None으로 대체
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

#LotFrontage
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#utility: 하나의 "NoSeWa"및 2 NA를 제외한 모든 데이터는 "AllPub"입니다.
df = df.drop(['Utilities'], axis=1)

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[col] = df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(0)

#확인필요
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col] = df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')

#MasVnrArea는 집들에 대한 고정 베니어가 없다는 것을 의미해서 결측치에 0을 넣어야함
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

#'RL'이 가장 일반적인 값이라 RL을 넣어줬다고함
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
#data description에 의하면 NA는 typical을 의미합니다
df["Functional"] = df["Functional"].fillna("Typ")
#하나의 NA 값이 있습니다. 이 기능은 대부분 'SBrkr'을 가지므로 결측값으로 설정할 수 있습니다.
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
#위와 마찬가지로 널 한갠데, 가장 빈번한 값을 넣어줌
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
#둘다 위와 동일
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
#가장 빈번한 데이터로 결측치 처리
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
#건물 등급 없음을 뜻해서 None으로 대체
df['MSSubClass'] = df['MSSubClass'].fillna("None")

print("------------------------결측치 처리 끝------------------------")
check_null = df.isna().sum()
print(check_null[check_null > 0])
print("-------------------------------------")


obj_df = df.select_dtypes(include='object')    # 카테고리형
num_df = df.select_dtypes(exclude='object')    # 수치형

print ("=======================feature engineering=======================")
df["OverallTotal"]=df["OverallQual"] + \
                   df["OverallCond"]
df["ExterTotal"]=df["ExterQual"] + \
                 df["ExterCond"]
df["BsmtTotal"]=df["BsmtQual"] + \
                df["BsmtCond"]
df["BathTotal"]=df["FullBath"] + \
                df["HalfBath"] * 0.5 + \
                df["BsmtFullBath"] + \
                df["BsmtHalfBath"] * 0.5
df["GarageTotal"]=df["GarageQual"] + \
                  df["GarageCond"]

df["FlrSFTotal"]=df["1stFlrSF"] + \
                 df["2ndFlrSF"]

df["AreaPerCar"]=df["GarageArea"] / \
                 df["GarageCars"]
df["AreaPerCar"]=df["AreaPerCar"].fillna(0)


#현관이나 베란다 등의 피트를 다 더한 것
df['Total_porch_sf'] = (df['OpenPorchSF']
                        + df['3SsnPorch']
                        + df['EnclosedPorch']
                        + df['ScreenPorch']
                        + df['WoodDeckSF']
                        )

# #값이 있는게 몇개 없는 feature들은 있는지 없는지로만 판단
df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df=df.drop(["OverallQual", "OverallCond","ExterQual","ExterCond","BsmtQual","BsmtCond", "BsmtFullBath","BsmtHalfBath","FullBath","HalfBath",
            "GarageQual","GarageCond","1stFlrSF","2ndFlrSF","LowQualFinSF", "GarageCars", "GarageArea", "PoolArea", "OpenPorchSF", "3SsnPorch", "EnclosedPorch",
            "ScreenPorch", "WoodDeckSF"],
           axis=1)

print(df.columns)


new_train = df[:train.shape[0]].copy()
new_test = df[train.shape[0]:].copy()

target = train['SalePrice']
target=target.reset_index(drop=True,inplace=False)
print ("=======================target=======================")
print(target)
print(target.shape)

add_col=target.values

new_train.loc[:,'SalePrice']=add_col
print(new_train.columns)

corrmat = new_train.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice'])>=0.3]


print(top_corr_features)
# heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(new_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()


# feature selection

selection_df = new_train[top_corr_features].copy()
selection_df['LotConfig'] = new_train['LotConfig']
selection_test_df = new_test[top_corr_features.drop(['SalePrice'])].copy()
print(selection_df)

target = selection_df['SalePrice']
df = selection_df.drop(['SalePrice'], axis=1).copy()

bestParam = {
        "scaler": ["standard", "robust", "minmax"],
        "encoder": ["labelEncoder", "oneHotEncoder"],
        "model": ["adaboost", "decisiontree", "bagging", "XGBoost", "gradient", "randomforest"]
}
encoding_cols = ['LotConfig']
scaling_cols = ['LotFrontage', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'TotalBsmtSF', 'GrLivArea', 'TotRmsAbvGrd', 'Fireplaces',
       'OverallTotal', 'BathTotal', 'FlrSFTotal', 'Total_porch_sf',
       'hasfireplace']

best_params, best_score = searchBest.bestSearchEncoding(bestParam, df, target, encoding_cols, scaling_cols)
print ("Best Combination, Score:", best_params, best_score)
