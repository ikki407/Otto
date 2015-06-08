import graphlab as gl
import math
import random
"""
対数変換
train_log = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv')
test_log = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv')
f = lambda x: math.log(x*x) if x >=2 else x
train_log.iloc[:,1:-1] = train_log.iloc[:,1:-1].applymap(f)
test_log.iloc[:,1:] = test_log.iloc[:,1:].applymap(f)

train_log.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_log.csv',index = False)
test_log.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_log.csv',index = False)
"""
train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_log.csv')
test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_log.csv')
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
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'step_size':0.4,
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

make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_log.csv')

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
submission_log

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission1_log

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 8,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission+submission1  

submission2_log

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 6,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'validation_set': None}

submission+submission1+submission2  0.44144

submission3_log
params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 6,
          'min_child_weight': 5,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': 1.0,
          'validation_set': None}

submission4_log
やばそう
params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 10,
          'min_child_weight': 5,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': 1.0,
          'validation_set': None}

sub_sub1+sub2+sub3+sub4 0.43911

submission5_log

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .6,
          'min_loss_reduction': 1,
          'column_subsample': .75,
          'validation_set': None}

submission6_log

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'validation_set': None}

submission7_log

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}

sub_sub1+sub2+sub3+sub5+sub6+sub7 0.43624
"""

