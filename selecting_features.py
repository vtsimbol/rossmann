import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDRegressor
from sklearn.cluster import KMeans


source_dataset_path = "D:\\Github\\rossmann\\train\\train_dataset.csv"
pre_train_path = "D:\\Github\\rossmann\\fit\\pre_train.csv"
pre_test_path = "D:\\Github\\rossmann\\fit\\pre_test.csv"

train_df = pd.read_csv(source_dataset_path, sep=',', index_col=False)

ratio = 0.9

def get_pre_train_and_test_datasets(train_dataset):
    train_df = train_dataset.copy()
    # Нормализация числовых полей
    train_df['StateHolidayDays'] = sklearn.preprocessing.normalize(train_df['StateHolidayDays'].values.reshape(-1, 1), axis=0)
    train_df['SchoolHolidayDays'] = sklearn.preprocessing.normalize(train_df['SchoolHolidayDays'].values.reshape(-1, 1), axis=0)
    train_df['Sales'] = sklearn.preprocessing.normalize(train_df['Sales'].values.reshape(-1, 1), axis=0)
    train_df['PromoDays'] = sklearn.preprocessing.normalize(train_df['PromoDays'].values.reshape(-1, 1), axis=0)
    train_df['Promo2DaysToNext'] = sklearn.preprocessing.normalize(train_df['Promo2DaysToNext'].values.reshape(-1, 1), axis=0)
    train_df['Promo2AllDays'] = sklearn.preprocessing.normalize(train_df['Promo2AllDays'].values.reshape(-1, 1), axis=0)
    train_df['Promo2Days'] = sklearn.preprocessing.normalize(train_df['Promo2Days'].values.reshape(-1, 1), axis=0)
    train_df['MeanByDay'] = sklearn.preprocessing.normalize(train_df['MeanByDay'].values.reshape(-1, 1), axis=0)
    train_df['Mean'] = sklearn.preprocessing.normalize(train_df['Mean'].values.reshape(-1, 1), axis=0)
    train_df['DayOfWeek'] = sklearn.preprocessing.normalize(train_df['DayOfWeek'].values.reshape(-1, 1), axis=0)
    train_df['DayOfMonth'] = sklearn.preprocessing.normalize(train_df['DayOfMonth'].values.reshape(-1, 1), axis=0)
    train_df['Customers'] = sklearn.preprocessing.normalize(train_df['Customers'].values.reshape(-1, 1), axis=0)
    train_df['CompetitionDistance'] = sklearn.preprocessing.normalize(train_df['CompetitionDistance'].values.reshape(-1, 1), axis=0)
    train_df['CompetitionDays'] = sklearn.preprocessing.normalize(train_df['CompetitionDays'].values.reshape(-1, 1), axis=0)
    # Работа с категориальными признаками
    objects = train_df.select_dtypes(include=[object])
    label_encoder = preprocessing.LabelEncoder()
    transf = objects.apply(label_encoder.fit_transform)
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(transf)
    labels = enc.transform(transf).toarray()
    # Заменяем/добавляем столбцы
    del train_df['Assortment']
    del train_df['StoreType']
    del train_df['StateHoliday']
    #train_df.drop(columns=['Assortment', 'StoreType', 'StateHoliday'], axis=1)
    for i in range(0, labels.shape[1]):
        train_df[f'x{i}'] = pd.Series(labels[:,i])
    # Перемешиваем
    indexes = np.arange(0, train_df.shape[0])
    np.random.shuffle(indexes)
    pos = int(train_df.shape[0] * ratio)
    train_indexes = indexes[:pos]
    test_indexes = indexes[pos:]
    train = train_df.loc[train_indexes]
    test = train_df.loc[test_indexes]
    return train, test

train, test = get_pre_train_and_test_datasets(train_df)
train.to_csv(pre_train_path, sep=',', index=False)
test.to_csv(pre_test_path, sep=',', index=False)

def fit_linear_regression(train, test):
    y = train['Sales']
    X_train = train.copy()
    X_test = test.copy()
    del X_train['Sales']
    del X_test['Sales']
    sgd = SGDRegressor(loss='huber', max_iter=1000)
    sgd.fit(X_train.values, y.values)
    return sgd.predict(X_train), sgd.predict(X_test)
    
lr_train, lr_test = fit_linear_regression(train, test)

train['x10'] = pd.Series(lr_train)
test['x10'] = pd.Series(lr_test)

def feature_visualization(train):
    y = train['Sales']
    X_train = train.copy()
    del X_train['Sales']
    import xgboost as xgb
    #gbm = xgb.XGBClassifier(silent=False, nthread=4, max_depth=10, n_estimators=800, subsample=0.5, learning_rate=0.03, seed=1337)
    #gbm = xgb.XGBClassifier(silent=False, nthread=2, max_depth=3, n_estimators=100, subsample=1, learning_rate=0.2, seed=1337)
    gbm = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', nthread=1)
    gbm.fit(X_train.values, y.values)
    print(y)
    bst = gbm.booster()
    imps = bst.get_fscore()
    return gbm
    
gbm = feature_visualization(train)
gbm.save_model('D:\\Github\\rossmann\\models\\gbm_model.bin')