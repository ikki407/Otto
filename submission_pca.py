from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
This script reads the data preforms PCA, and keeps only N of the best components
'''

from sklearn import decomposition
import pandas as pd
import os
import numpy as np
N = 92
print "reading train"
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')

print "reading test"
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')

rows = random.sample(train.index, len(train))
train = train.ix[rows]
rows = random.sample(test.index, len(test))
test = test.ix[rows]


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




##############################################################################


import graphlab as gl
import math
import random

train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_pca_92.csv')
test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_pca_92.csv')
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
          'max_iterations':20,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}
#パラメータのdefault調べる  stepsize = 
train = shuffle(train)

# Check performance on internal validation set
#一旦、valでだいたいの値を評価してから、全trainデータで評価する。
tr, va = train.random_split(.8)
m = gl.boosted_trees_classifier.create(tr, **params)

'''
xgboostに書き換える
tr, va = train.random_split(.8)
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob' }
# specify validations set to watch performance
watchlist  = [(va,'eval'), (tr,'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
'''
print evaluate_logloss(m, tr, va)

# Make final submission by using full training set
m = gl.boosted_trees_classifier.create(train, **params)
'''
xgboostに書き換える
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob' }
# specify validations set to watch performance
watchlist  = [(test,'eval'), (train,'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
'''

make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/sub1_sd_pca.csv')


