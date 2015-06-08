from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import math
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from operator import itemgetter
# Utility function to report best scores
def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

X, target, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count.csv')
test, ids = load_test_data('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count.csv', scaler)

from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# clf = XGBClassifier()
from scipy.stats import randint as sp_randint

# print help(clf)
# param_dist = {"n_estimators": [100, 200],
#               "max_depth": [None, 8, 10, 12, 20],
#               'learning_rate': [0.1],
#               # "max_features": sp_randint(1, 11),
#               # "min_samples_split": sp_randint(1, 11),
#               # "min_samples_leaf": sp_randint(1, 11),
#               }


random_state = 42
n_folds = 2

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_dist = dict(gamma=gamma_range, C=C_range)

clf = SVC(probability=True, cache_size=2048)
# searhc = GridSearchCV(clf, param_dist, random_state=42, cv=2, scoring='log_loss', verbose=3, n_jobs=-1)
skf = cross_validation.StratifiedKFold(target,
                                       n_folds=n_folds,
                                       random_state=random_state)
# cv = StratifiedShuffleSplit(target, n_iter=5, test_size=0.5, random_state=42)
random_search = RandomizedSearchCV(clf,
                                   param_dist,
                                   random_state=random_state,
                                   cv=skf,
                                   scoring='log_loss',
                                   verbose=3,
                                   n_jobs=-1,
                                   n_iter=100)

# cv = StratifiedShuffleSplit(target, n_iter=5, test_size=0.3, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_dist, cv=cv, n_jobs=1, verbose=3)
#
fit = random_search.fit(X, target)
report(fit.grid_scores_)

from sklearn.multiclass import OneVsRestClassifier
clf3 = OneVsRestClassifier(SVC(C=5, cache_size=2048), n_jobs=-1)
scores = cross_validation.cross_val_score(estimator=eclf,X=X,y=target,cv=5,scoring='log_loss', n_jobs = -1,verbose=verbose)



import pandas as pd
import numpy as np
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv')
target = train['target']
Id_tr = train['id']
Id_te = test['id']
X = pd.concat((train[[col for col in train.columns if col.startswith("feat")]], test))
count_lk = {col: X.groupby(col).aggregate({"id": "count"}).to_dict()["id"] for col in X.columns}

y_lk = {t: i for i, t in enumerate(train.target.unique())}
y = np.array([y_lk[t] for t in train.pop("target")])
ids = train.pop("id")
test_ids = test.pop("id")
for col in train.iloc[:,0:93].columns:
    train[col + "_log"] = [np.log(1+x) if x else 0 for x in train[col]]
    #train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    test[col + "_log"] = [np.log(1+x) if x else 0 for x in test[col]]
    #test[col + "_count"] = ([count_lk[col][x] for x in test[col]])
for col in train.iloc[:,0:93].columns:
    #train[col + "_log"] = [np.log(1+x) if x else 0 for x in train[col]]
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    #test[col + "_log"] = [np.log(1+x) if x else 0 for x in test[col]]
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])

train['feat_row_sum_orig'] = train.iloc[:,0:93].sum(1)
train['feat_row_std_orig'] = train.iloc[:,0:93].std(1)
train['feat_row_sum_log'] = train.iloc[:,93:186].sum(1)
train['feat_row_std_log'] = train.iloc[:,93:186].std(1)
train['feat_row_sum_count'] = train.iloc[:,186:279].sum(1)
train['feat_row_std_count'] = train.iloc[:,186:279].std(1)

test['feat_row_sum_orig'] = test.iloc[:,0:93].sum(1)
test['feat_row_std_orig'] = test.iloc[:,0:93].std(1)
test['feat_row_sum_log'] = test.iloc[:,93:186].sum(1)
test['feat_row_std_log'] = test.iloc[:,93:186].std(1)
test['feat_row_sum_count'] = test.iloc[:,186:279].sum(1)
test['feat_row_std_count'] = test.iloc[:,186:279].std(1)

# Create is_zero binaries
orig_feats = list(train.iloc[:,0:93].columns.values)
is_zero_nms = list(train.iloc[:,0:93].columns.values)
for index in range(0, len(orig_feats)):
    is_zero_nms[index] = "is_zero_" + is_zero_nms[index]
train[is_zero_nms] = (train[orig_feats] > 0).applymap(int)
# Count times non-zero
train['feat_cnt_non_zero_orig'] = train.ix[:, is_zero_nms].sum(1)
    
orig_feats = list(test.iloc[:,0:93].columns.values)
is_zero_nms = list(test.iloc[:,0:93].columns.values)
for index in range(0, len(orig_feats)):
    is_zero_nms[index] = "is_zero_" + is_zero_nms[index]
test[is_zero_nms] = (test[orig_feats] > 0).applymap(int)
# Count times non-zero
test['feat_cnt_non_zero_orig'] = test.ix[:, is_zero_nms].sum(1)


train = pd.concat([Id_tr,train,target],axis=1)
test = pd.concat([Id_te,test],axis=1)
train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_sammary.csv',index = False)
test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_sammary.csv',index = False)



