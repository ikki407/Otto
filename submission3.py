import graphlab as gl
import math
import random
"""
#Standardize
train_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv')
test_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv')

mean_tr = train_sd.iloc[:,1:-1].mean(axis=0)
std_tr = train_sd.iloc[:,1:-1].std(axis=0)
train_sd.iloc[:,1:-1] = (train_sd.iloc[:,1:-1] - mean_tr)/std_tr

mean_te = test_sd.iloc[:,1:].mean(axis=0)
std_te = test_sd.iloc[:,1:].std(axis=0)
test_sd.iloc[:,1:] = (test_sd.iloc[:,1:] - mean_te)/std_te

train_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv',index = False)
test_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv',index = False)
"""
train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')
test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')
del train['id']

def make_submission(m, test, filename):
    preds = m.predict_topk(test, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds = preds.sort('id')
    preds.save(filename)

def multiclass_logloss(model, test):
    preds = model.predict_topk(test, output_type='probability', k=9)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    preds['target'] = test['target']
    neg_log_loss = 0
    for row in preds:
        label = row['target']
        neg_log_loss += - math.log(row[label])
    return  neg_log_loss / preds.num_rows()

def shuffle(sf):
    sf['_id'] = [random.random() for i in xrange(sf.num_rows())]
    sf = sf.sort('_id')
    del sf['_id']
    return sf

def evaluate_logloss(model, train, valid):
    return {'train_logloss': multiclass_logloss(model, train),
            'valid_logloss': multiclass_logloss(model, valid)}

params = {'target': 'target',
          'max_iterations': 350,
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}


#パラメータのdefault調べる  stepsize = 
train = shuffle(train)

# Check performance on internal validation set
#一旦、valでだいたいの値を評価してから、全trainデータで評価する。
tr, va = train.random_split(.8)
m = gl.boosted_trees_classifier.create(tr, **params)

print evaluate_logloss(m, tr, va)

# Make final submission by using full training set
m = gl.boosted_trees_classifier.create(train, **params)

make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission15_sd.csv')

#バギング
import pandas as pd
sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_log.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission1_log.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission2_log.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission3_log.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_log.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_log.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_log.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7)/7
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sub_sub1_sub2_sub3_sub4_sub5_sub6_log.csv',index = False)




"""
submission_sd

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission1_sd 

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 8,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission+submission1  

submission2_sd 0.
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 6,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'validation_set': None}

submission+submission1+submission2  0.44108

submission3_sd
params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 6,
          'min_child_weight': 5,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': 1.0,
          'validation_set': None}

submission4_sd
params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 10,
          'min_child_weight': 5,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': 1.0,
          'validation_set': None}

sub_sub1+sub2+sub3+sub4 0.43846

submission5_sd 
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .6,
          'min_loss_reduction': 1,
          'column_subsample': .75,
          'validation_set': None}

submission6_sd

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'validation_set': None}

sub_sub1_sub2_sub3_sub5_sub6 0.43701

submission7_sd

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}

sub_sub1_sub2_sub3_sub5_sub6_sub7 0.43609

submission8_sd
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}

sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8 0.43524

submission9_sd 
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}

submission10_sd
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 14,
          'min_child_weight': 4,
          'row_subsample': 1.,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}

submission11_sd
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 17,
          'min_child_weight': 5,
          'row_subsample': 1.,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}

submission12_sd
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 3,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission13_sd 
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 4,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}

submission14_sd
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'step_size': 0.2,
          'class_weights':'auto',
          'validation_set': None}

submission15_sd
params = {'target': 'target',
          'max_iterations': 350,
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_1_3_4
0.42431

sd_sub_sub5_sub6_sub7_sub8_sub9_sub12_sub13_neural_1_3_4
0.43107

sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission1.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission3.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission4.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission12_sd.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission13_sd.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11)/11
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sd_sub_sub5_sub6_sub7_sub8_sub9_neural_1_3_4_8vs2.csv',index = False)


"""




from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
This script reads the data preforms PCA, and keeps only N of the best components
'''

from sklearn import decomposition
import pandas as pd
import os

N = 40
print "reading train"
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')

print "reading test"
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')

target = train["target"]
train_id = train["id"]
test_id = test["id"]

train = train.drop(["id", "target"], 1)
test = test.drop("id", 1)

pca = decomposition.PCA(n_components=N)
print "fitting"
pca.fit(train)
print "transforming train"
x = pd.DataFrame(pca.transform(train))

x["id"] = train_id
x["target"] = target

x.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_pca_{N}.csv'.format(N=N))

print "transforming test"
y = pd.DataFrame(pca.transform(test))
y["id"] = test_id
y.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_pca_{N}.csv'.format(N=N))


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
for col in train.columns:
    train[col + "_log"] = [np.log(1+x) if x else 0 for x in train[col]]
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    test[col + "_log"] = [np.log(1+x) if x else 0 for x in test[col]]
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])

train = pd.concat([Id_tr,train,target],axis=1)
test = pd.concat([Id_te,test],axis=1)
train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count.csv',index = False)
test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count.csv',index = False)


train_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count.csv')
test_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count.csv')

train_sd['target'] = target
train_sd['id'] = Id_tr
test['id'] Id_te
train_sd = pd.concat([Id_tr,train_sd,target],axis=1)
test_sd = pd.concat([Id_te,test_sd],axis=1)
mean_tr = train_sd.iloc[:,1:-1].mean(axis=0)
std_tr = train_sd.iloc[:,1:-1].std(axis=0)
train_sd.iloc[:,1:-1] = (train_sd.iloc[:,1:-1] - mean_tr)/std_tr

mean_te = test_sd.iloc[:,1:].mean(axis=0)
std_te = test_sd.iloc[:,1:].std(axis=0)
test_sd.iloc[:,1:] = (test_sd.iloc[:,1:] - mean_te)/std_te

train_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_count.csv',index = False)
test_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_count.csv',index = False)


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
for col in train.columns:
    train[col + "_log"] = [np.log(1+x) if x else 0 for x in train[col]]
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    train[col + "_count"] = [np.log(1+x) if x else 0 for x in train[col + "_count"]]
    test[col + "_log"] = [np.log(1+x) if x else 0 for x in test[col]]
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])
    test[col + "_count"] = [np.log(1+x) if x else 0 for x in test[col + "_count"]]

train = pd.concat([Id_tr,train,target],axis=1)
test = pd.concat([Id_te,test],axis=1)
train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_log.csv',index = False)
test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_log.csv',index = False)


'''
count_only
'''
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
for col in train.columns:
    #train[col + "_log"] = [np.log(1+x) if x else 0 for x in train[col]]
    train[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    #test[col + "_log"] = [np.log(1+x) if x else 0 for x in test[col]]
    test[col + "_count"] = ([count_lk[col][x] for x in test[col]])

train = pd.concat([Id_tr,train,target],axis=1)
test = pd.concat([Id_te,test],axis=1)
train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_only.csv',index = False)
test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_only.csv',index = False)


train_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_only.csv')
test_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_only.csv')

#train_sd['target'] = target
#train_sd['id'] = Id_tr
#test['id'] = Id_te
#train_sd = pd.concat([Id_tr,train_sd,target],axis=1)
#test_sd = pd.concat([Id_te,test_sd],axis=1)
mean_tr = train_sd.iloc[:,1:-1].mean(axis=0)
std_tr = train_sd.iloc[:,1:-1].std(axis=0)
train_sd.iloc[:,1:-1] = (train_sd.iloc[:,1:-1] - mean_tr)/std_tr

mean_te = test_sd.iloc[:,1:].mean(axis=0)
std_te = test_sd.iloc[:,1:].std(axis=0)
test_sd.iloc[:,1:] = (test_sd.iloc[:,1:] - mean_te)/std_te

train_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_count_only.csv',index = False)
test_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_count_only.csv',index = False)



from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from pandas import DataFrame
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

tr_te =  pd.concat([train,test],axis=0)
mean_tr_te = np.mean(tr_te)
std_tr_te = np.std(tr_te)

tr_te_sd = (tr_te - mean_tr_te)/std_tr_te
train_sd = tr_te_sd.iloc[:len(train),:]
test_sd = tr_te_sd.iloc[len(train):,:]

wpca = PCA(whiten=True)
wpca.fit(np.log(tr_te+1))
train_wpca = DataFrame(wpca.transform(np.log(tr_te+1))).iloc[:len(train),:]
test_wpca = DataFrame(wpca.transform(np.log(tr_te+1))).iloc[len(train):,:]

train_count = DataFrame()
test_count = DataFrame()
for col in train.columns:
    #train[col + "_log"] = [np.log(1+x) if x else 0 for x in train[col]]
    train_count[col + "_count"] = ([count_lk[col][x] for x in train[col]])
    train_count[col + "_count"] = [np.log(1+x) if x else 0 for x in train_count[col + "_count"]]
    #test[col + "_log"] = [np.log(1+x) if x else 0 for x in test[col]]
    test_count[col + "_count"] = ([count_lk[col][x] for x in test[col]])
    test_count[col + "_count"] = [np.log(1+x) if x else 0 for x in test_count[col + "_count"]]

test_wpca.index = test.index
train_wpca.index = train.index
train = pd.concat([Id_tr,train_sd,train_count,train_wpca,target],axis=1)
test = pd.concat([Id_te,test_sd,test_count,test_wpca],axis=1)
train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_pca.csv',index = False)
test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_pca.csv',index = False)


train_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_pca.csv')
test_sd = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_pca.csv')

#train_sd['target'] = target
#train_sd['id'] = Id_tr
#test['id'] = Id_te
#train_sd = pd.concat([Id_tr,train_sd,target],axis=1)
#test_sd = pd.concat([Id_te,test_sd],axis=1)
mean_tr = train_sd.iloc[:,1:-1].mean(axis=0)
std_tr = train_sd.iloc[:,1:-1].std(axis=0)
train_sd.iloc[:,1:-1] = (train_sd.iloc[:,1:-1] - mean_tr)/std_tr

mean_te = test_sd.iloc[:,1:].mean(axis=0)
std_te = test_sd.iloc[:,1:].std(axis=0)
test_sd.iloc[:,1:] = (test_sd.iloc[:,1:] - mean_te)/std_te

train_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_count_pca.csv',index = False)
test_sd.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_count_pca.csv',index = False)





from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
poly.fit_transform(train.head().iloc[:,1:-1])

import pandas as pd
import numpy as np
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_only.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_only.csv')

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
train = poly.fit_transform(train.iloc[:,1:-1])
test = poly.fit_transform(test.iloc[:,1:-1])

train=pd.DataFrame(train)
test=pd.DataFrame(test)
from pandas.io.pytables import HDFStore
store = HDFStore('/Users/IkkiTanaka/Documents/kaggle/Otto/store.h5')
store['train'] = train

train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_only_qf.csv',index = False)
test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_only_qf.csv',index = False)

train.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_only_qf.dump')
test.save('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_only_qf.dump')
