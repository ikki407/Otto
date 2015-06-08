import graphlab as gl
import math
import random
import pandas as pd
train_log = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv')
test_log = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv')
f = lambda x: math.log(x*x) if x >=2 else x
train_log.iloc[:,1:-1] = train_log.iloc[:,1:-1].applymap(f)
test_log.iloc[:,1:] = test_log.iloc[:,1:].applymap(f)

train_log.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_log.csv',index = False)
test_log.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_log.csv',index = False)

train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv')
test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv')
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
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          'validation_set': None}

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

make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission11.csv')
'''
# this is prediction
preds = bst.predict(test)
'''

#バギング
import pandas as pd
sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission1.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission2.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission3.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8)/8
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8.csv',index = False)




"""
submission  0.45612

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission1 0.45670

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 8,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

submission+submission1  0.44421

submission2 0.45919
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 6,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'validation_set': None}

submission+submission1+submission2  0.44108

submission3 0.45994
params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 6,
          'min_child_weight': 5,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': 1.0,
          'validation_set': None}

submission4 0.46373
params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 10,
          'min_child_weight': 5,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': 1.0,
          'validation_set': None}

sub_sub1+sub2+sub3+sub4 0.43846

submission5 
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .6,
          'min_loss_reduction': 1,
          'column_subsample': .75,
          'validation_set': None}

submission6

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'validation_set': None}

sub_sub1_sub2_sub3_sub5_sub6 0.43701
submission7

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

submission8 
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

submission9 
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}

submission10
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 14,
          'min_child_weight': 4,
          'row_subsample': 1.,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}

submission11
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 17,
          'min_child_weight': 5,
          'row_subsample': 1.,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}


"""

from collections import Counter
cnt = Counter()
for i in train['target']:
    cnt[i] += 1
print cnt
cnt_final = Counter()
for i in pred_counter:
    cnt_final[i] += 1
print cnt_final

def make_submission_top1(m, test):
    preds = m.predict_topk(test, output_type='rank', k=1)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'rank'], 'rank').unpack('rank', '')
    preds = preds.sort('id')
    return preds

pred_counter = make_submission_top1(m, test)
pred_counter1 = pred_counter.to_dataframe()
pred_counter1 = pred_counter1.fillna(1)

results = {}
for columns in [u'Class_1', u'Class_2', u'Class_3', u'Class_4', u'Class_5', u'Class_6', u'Class_7', u'Class_8', u'Class_9']:
    cnt_pred = Counter()
    for i in pred_counter1[columns]:
        cnt_pred[i] += 1
    results[columns] = cnt_pred[0]
#print results
for k, v in sorted(results.items(), key=lambda x:x[1]):
    print k, v
len(test)/len(train)








from sklearn import feature_extraction
from pandas import DataFrame
import pandas as pd
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')
del train['id']

train_plus = DataFrame()
pre_index = 0
for Class in [u'Class_1', u'Class_2', u'Class_3', u'Class_4', u'Class_5', u'Class_6', u'Class_7', u'Class_8', u'Class_9']:
    train_ = train[(train['target']==Class)]
    train_des = train_.describe()
    train_plus2 = DataFrame()
    print '1'
    for i in train.columns[:-1]:
        train__ = DataFrame(train_des[i]).T
        train__.columns = i + '_' + train__.columns
        train__.index = [0]
        train_plus2 = pd.concat([train_plus2,train__], axis=1)
    train_plus3 = DataFrame()
    print '2'
    for k in xrange(0,len(train_)):
        train_plus3 = pd.concat([train_plus3,train_plus2], axis=0)
        print k
    train_plus3.index = range(pre_index ,(pre_index+len(train_)))
    pre_index = len(train_)
    train_plus = pd.concat([train_plus, train_plus3], axis=0)
    print '3'
train = pd.concat([train,train_plus],axis=1)

train.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv'


