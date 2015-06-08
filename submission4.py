from graphlab.toolkits.feature_engineering import *
import graphlab as gl
import math
import random
from pandas import DataFrame
import pandas as pd
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
del test['id']
'''
In [1]: range(0,61800,10000)
Out[1]: [0, 10000, 20000, 30000, 40000, 50000, 60000]
端作れ！！
'''
train = train.to_dataframe()
train_qf = DataFrame()
for h in xrange(0,len(train),10000):
    if h > len(train)-10000:
        train_ = train[h:]
    else:
        train_ = train[h:(h+10000)]
    train_qf_ = DataFrame()
    for i in xrange(0,len(train_),1000):
        if i > len(train_)-1000:
            train__ = train_[i:]
        else:
            train__ = train_[i:(i+1000)]
        train__ = gl.SFrame(train__)
        quadratic = gl.feature_engineering.create(train__, QuadraticFeatures())
        #a=quadratic.fit(train_)
        train__=quadratic.fit_transform(train__)
        train__ = train__.to_dataframe()

        if i > len(train_)-1000:
            train__.index = range(i,len(train_))
        else:
            train__.index = range(i,(i+1000))
        train_qf_ = pd.concat([train_qf_,train__],axis=0)
        print '内側変換%i' % i
    if h > len(train)-10000:
        train_qf_.index = range(h,len(train))
    else:
        train_qf_.index = range(h,(h+10000))
    #test_qf = pd.concat([test_qf,test_qf_],axis=0)
    print '外側変換%i' % h
    #output_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_%i.csv' % h
    #test_qf_.to_csv(output_name ,index = False)
    train_qf_.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_%i.dump' % h)
del train_qf_, train_

#train_qf.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf.dump')
#pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf.dump')

test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')
del test['id']

test = test.to_dataframe()
test_qf = DataFrame()
for h in xrange(0,len(test),10000):
    if h > len(test)-10000:
        test_ = test[h:]
    else:
        test_ = test[h:(h+10000)]
    test_qf_ = DataFrame()
    for i in xrange(0,len(test_),1000):
        if i > len(test_)-1000:
            test__ = test_[i:]
        else:
            test__ = test_[i:(i+1000)]
        test__ = gl.SFrame(test__)
        quadratic = gl.feature_engineering.create(test__, QuadraticFeatures())
        #a=quadratic.fit(train_)
        test__=quadratic.fit_transform(test__)
        test__ = test__.to_dataframe()

        if i > len(test_)-1000:
            test__.index = range(i,len(test_))
        else:
            test__.index = range(i,(i+1000))
        test_qf_ = pd.concat([test_qf_,test__],axis=0)
        print '内側変換%i' % i
    if h > len(test)-10000:
        test_qf_.index = range(h,len(test))
    else:
        test_qf_.index = range(h,(h+10000))
    #test_qf = pd.concat([test_qf,test_qf_],axis=0)
    print '外側変換%i' % h
    #output_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_%i.csv' % h
    #test_qf_.to_csv(output_name ,index = False)
    test_qf_.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_%i.dump' % h)
del test_qf_, test_


a = pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_%i.dump' % h)
test0 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_0.csv')
test1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_10000.csv')
test2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_20000.csv')
test3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_30000.csv')
test4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_40000.csv')
test5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_50000.csv')
test6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_60000.csv')
test7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_70000.csv')
test8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_80000.csv')
test9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_90000.csv')
test10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_100000.csv')
test11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_110000.csv')
test12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_120000.csv')
test13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_130000.csv')
test14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_140000.csv')

test_qf = pd.concat([test0,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,tes11,test12,test13,test14],axis=0)

test_qf.index = range(1,(len(test_qf)+1))
test_qf.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf.csv',index=False)

#test_qf.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf.csv',index = False)
'''
ニューラルネットのためのDataFrameづくり

train_sd_qf_net = DataFrame((train_qf['quadratic_features']).tolist(),index=range(0,len(train_qf)))

test_sd_qf_net = DataFrame((test_qf['quadratic_features']).tolist(),index=range(0,len(test_qf)))

train_sd_qf_net = DataFrame()

for i in xrange(0,len(train_qf),1000):
    if i > len(train_qf)-1000:
        train_qf_ = train_qf[i:]
    else:
        train_qf_ = train_qf[i:(i+1000)]
    train_qf_ = DataFrame((train_qf_['quadratic_features']).tolist(),index=range(0,len(train_qf_)))
    if i > len(train)-1000:
        train_qf_.index = range(i,len(train))
    else:
        train_qf_.index = range(i,(i+1000))

    train_sd_qf_net = pd.concat([train_sd_qf_net,train_qf_],axis=0)
    print i


test_sd_qf_net = DataFrame()
for i in xrange(0,len(test_qf),1000):
    if i > len(test_qf)-1000:
        test_qf_ = test_qf[i:]
    else:
        test_qf_ = test_qf[i:(i+1000)]
    test_qf_ = DataFrame((test_qf_['quadratic_features']).tolist(),index=range(0,len(test_qf_)))
    if i > len(test)-1000:
        test_qf_.index = range(i,len(test))
    else:
        test_qf_.index = range(i,(i+1000))

    test_sd_qf_net = pd.concat([test_sd_qf_net,test_qf_],axis=0)
    print i
'''
trian_qf = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf.csv')
train_sd_qf_net = DataFrame()
for h in xrange(0,len(train_qf),10000):
    if h > len(train_qf)-10000:
        train_qf_ = train_qf[h:]
    else:
        train_qf_ = train_qf[h:(h+10000)]
    train_sd_qf_net_ = DataFrame()
    for i in xrange(0,len(train_qf_),1000):
        if i > len(train_qf_)-1000:
            train_qf__ = train_qf_[i:]
        else:
            train_qf__ = train_qf_[i:(i+1000)]
        train_qf__ = DataFrame((train_qf__['quadratic_features']).tolist(),index=range(0,len(train_qf__)))
        if i > len(train_qf_)-1000:
            train_qf__.index = range(i,len(train_qf_))
        else:
            train_qf__.index = range(i,(i+1000))

        train_sd_qf_net_ = pd.concat([train_sd_qf_net_,train_qf__],axis=0)
        print '内側変換%i' % i
    if h > len(train_qf)-10000:
        train_sd_qf_net_.index = range(h,len(train_qf))
    else:
        train_sd_qf_net_.index = range(h,(h+10000))

    train_sd_qf_net = pd.concat([train_sd_qf_net,train_sd_qf_net_],axis=0)
    print '外側変換%i' % h
train_sd_qf_net.index = train.index
x = pd.concat([train,train_sd_qf_net],axis=1)
train_sd_qf_net.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net.csv',index = False)

del train_sd_qf_net

#test_qf = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf.csv')
#test_sd_qf_net = DataFrame()
'''
test
'''
for h in xrange(0,len(test),10000):
    #if h > len(test_qf)-10000:
    #    test_qf_ = test_qf[h:]
    #else:
    #    test_qf_ = test_qf[h:(h+10000)]
    test_qf_ = pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_%i.dump' % h)
    test_sd_qf_net_ = DataFrame()
    for i in xrange(0,len(test_qf_),1000):
        if i > len(test_qf_)-1000:
            test_qf__ = test_qf_[i:]
        else:
            test_qf__ = test_qf_[i:(i+1000)]
        test_qf__ = DataFrame((test_qf__['quadratic_features']).tolist(),index=range(0,len(test_qf__)))
        if i > len(test_qf_)-1000:
            test_qf__.index = range(i,len(test_qf_))
        else:
            test_qf__.index = range(i,(i+1000))

        test_sd_qf_net_ = pd.concat([test_sd_qf_net_,test_qf__],axis=0)
        print '内側変換%i' % i
    if h > len(test)-10000:
        test_sd_qf_net_.index = range(h,len(test))
    else:
        test_sd_qf_net_.index = range(h,(h+10000))
    test_sd_qf_net_.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_%i.dump' % h)

    #test_sd_qf_net = pd.concat([test_sd_qf_net,test_sd_qf_net_],axis=0)
    print '外側変換%i' % h
#test_sd_qf_net.index = test.index
#x = pd.concat([test,test_sd_qf_net],axis=1)
#test_sd_qf_net.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net.csv',index = False)
'''
train
'''
for h in xrange(0,len(train),10000):
    #if h > len(test_qf)-10000:
    #    test_qf_ = test_qf[h:]
    #else:
    #    test_qf_ = test_qf[h:(h+10000)]
    train_qf_ = pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_%i.dump' % h)
    train_sd_qf_net_ = DataFrame()
    for i in xrange(0,len(train_qf_),1000):
        if i > len(train_qf_)-1000:
            train_qf__ = train_qf_[i:]
        else:
            train_qf__ = train_qf_[i:(i+1000)]
        train_qf__ = DataFrame((train_qf__['quadratic_features']).tolist(),index=range(0,len(train_qf__)))
        if i > len(train_qf_)-1000:
            train_qf__.index = range(i,len(train_qf_))
        else:
            train_qf__.index = range(i,(i+1000))

        train_sd_qf_net_ = pd.concat([train_sd_qf_net_,train_qf__],axis=0)
        print '内側変換%i' % i
    if h > len(train)-10000:
        train_sd_qf_net_.index = range(h,len(train))
    else:
        train_sd_qf_net_.index = range(h,(h+10000))
    train_sd_qf_net_.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_%i.dump' % h)

    #test_sd_qf_net = pd.concat([test_sd_qf_net,test_sd_qf_net_],axis=0)
    print '外側変換%i' % h








'''
train_sd_qf_net.dump作成
'''
tr0=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_0.dump')
tr1=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_10000.dump')
tr2=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_20000.dump')
tr3=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_30000.dump')
tr4=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_40000.dump')
tr5=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_50000.dump')
tr6=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_net_60000.dump')

tr_sd_qf_net = pd.concat([tr0,tr1,tr2,tr3,tr4,tr5,tr6],axis=0)
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')
#tr_sd_qf_net = pd.concat([train,tr_sd_qf_net],axis=1)
#tr_sd_qf_net.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/tr_sd_qf_net.csv',index=False)
train = pd.concat([train,tr_sd_qf_net],axis=1)





te0=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_0.dump')
te1=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_10000.dump')
te2=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_20000.dump')
te3=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_30000.dump')
te4=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_40000.dump')
te5=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_50000.dump')
te6=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_60000.dump')
te7=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_70000.dump')
te8=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_80000.dump')
te9=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_90000.dump')
te10=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_100000.dump')
te11=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_110000.dump')
te12=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_120000.dump')
te13=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_130000.dump')
te14=pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf_net_140000.dump')

te_sd_qf_net = pd.concat([te0,te1,te2,te3,te4,te5,te6,te7,te8,te9,te10,te11,te12,te13,te14],axis=0)
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')
#te_sd_qf_net = pd.concat([test,tr_sd_qf_net],axis=1)
#te_sd_qf_net.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/te_sd_qf_net.csv',index=False)
test = pd.concat([test,te_sd_qf_net],axis=1)

train = gl.SFrame(train)
test = gl.SFrame(test)

#train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf.csv')
#test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_qf.csv')
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
          'max_iterations': 300,
          'max_depth': 17,
          'min_child_weight': 5,
          'row_subsample': 1.,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}
#'convergence_threshold': 0.01
#パラメータのdefault調べる  stepsize = 
train = shuffle(train)

# Check performance on internal validation set
#一旦、valでだいたいの値を評価してから、全trainデータで評価する。
tr, va = train.random_split(.8)
#m = gl.boosted_trees_classifier.create(tr, **params)
m = gl.boosted_trees_classifier.create(tr, **params)

#tr, va = b.random_split(.8)
#m = gl.boosted_trees_classifier.create(tr, **params)

'''
def shuffle(sf):
    sf['_id'] = [random.random() for i in xrange(len(sf))]
    sf = sf.sort('_id')
    del sf['_id']
    return sf

xgboostに書き換える
import xgboost as xgb
train = shuffle(train)
#train0 = train.values
#train0 = random.shuffle(train0)
tr, va = train[:59000],train[59000:]
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob' }
# specify validations set to watch performance
watchlist  = [(va,'eval'), (tr,'train')]
num_round = 2
bst = xgb.train(param, train, num_round, watchlist)
'''
print evaluate_logloss(m, tr, va)

# Make final submission by using full training set
#m = gl.boosted_trees_classifier.create(train, **params)
print 'featureの数 94*93/2+93=4464'
m = gl.boosted_trees_classifier.create(train, **params)

#BoostedTreesClassifier, graphlab.logistic_classifier.LogisticClassifier, graphlab.svm_classifier.SVMClassifier, graphlab.neuralnet_classifier.NeuralNetClassifier

'''
xgboostに書き換える
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob' }
# specify validations set to watch performance
watchlist  = [(test,'eval'), (train,'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
'''

submission = DataFrame()
for h in xrange(0,len(test),10000):#ここかえる!!
    test1 = pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_qf_%i.dump' % h)
    if h > len(test)-10000:
        test1['id'] = range(h,len(test))
    else:
        test1['id'] = range(h:(h+10000))

    make_submission(m, test1, '/Users/IkkiTanaka/Documents/kaggle/Otto/sub_gbdt_sd_qf_%i.csv' % h)
    test_ = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sub_gbdt_sd_qf_%i.csv' % h)
    submission = pd.concat([submission, test_],axis=0)
submission.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sub1_gbdt_sd_qf_.csv', index=False)

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
submission_lr
params = {'target': 'target',
          'max_iterations': 250,
          'l2_penalty': 0.5,
          'l1_penalty': 0.5,
          'solver': 'auto',
          'lbfgs_memory_level' : 400,
          'step_size': 0.3,
          'validation_set': None }

submission1_lr
params = {'target': 'target',
          'max_iterations': 250,
          'l2_penalty': 0.3,
          'l1_penalty': 0.0,
          'solver': 'auto',
          'lbfgs_memory_level' : 400,
          'step_size': 0.3,
          'validation_set': None }


sd_sub_sub5_sub6_sub7_sub8_sub9_neural_1_3_4
0.42431

sd_sub_sub5_sub6_sub7_sub8_sub9_sub12_sub13_neural_1_3_4
0.43107

sd_sub_sub5_sub6_sub7_sub8_sub9_lr_sub1_neural_1_3_4
0.43040
sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission1.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission3.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission4.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission1_lr.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10)/10
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sd_sub_sub5_sub6_sub7_sub8_sub9_lr_sub1_neural_1_3_4.csv',index = False)


"""

install_name_tool -change "/usr/local/lib/libvw.0.dylib" /usr/local/lib/libvw.0.dylib /usr/local/bin/vw




###############
#SVM
###############


params = {'target': 'target',
          'penalty' : 1.0,
          'solver': 'auto',
          'feature_rescaling' : False,
          'convergence_threshold' : 0.01,
          'max_iterations' :200,
          'lbfgs_memory_level' : 300,
          'validation_set': None }        

#パラメータのdefault調べる  stepsize = 
train = shuffle(train)

# Check performance on internal validation set
#一旦、valでだいたいの値を評価してから、全trainデータで評価する。
tr, va = train.random_split(.8)
#m = gl.boosted_trees_classifier.create(tr, **params)
m = gl.svm_classifier.create(tr, **params)

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
#m = gl.boosted_trees_classifier.create(train, **params)
m = gl.svm_classifier.create(train, **params)

#BoostedTreesClassifier, graphlab.logistic_classifier.LogisticClassifier, graphlab.svm_classifier.SVMClassifier, graphlab.neuralnet_classifier.NeuralNetClassifier

'''
xgboostに書き換える
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob' }
# specify validations set to watch performance
watchlist  = [(test,'eval'), (train,'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
'''

make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission_svc.csv')



train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_log.csv',sep=',')
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv',sep=',')


#シャッフルする！
#VW用に変換
import sys
sys.path.append('/Users/IkkiTanaka/Documents/kaggle/Otto/')
import csv_to_vw_otto
csv_to_vw_otto.csv_to_vw('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv','/Users/IkkiTanaka/Documents/kaggle/Otto/train.vw',train=True)
csv_to_vw_otto.csv_to_vw('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv','/Users/IkkiTanaka/Documents/kaggle/Otto/test.vw',train=False)

cd /Users/IkkiTanaka/Documents/kaggle/Otto/
#subprocessモジュールを使用してterminalでVWを実行
import subprocess
command_train = 'vw train.vw -k -c -f train.model.vw --oaa 9 --loss_function logistic --passes 20 -l 0.5 -b 30 --nn 50 --holdout_period 5 -q ii'
command_test = 'vw test_vw3.vw -t -i train.model3.vw -p test3.txt'

subprocess.check_output(command_train, shell=True)
subprocess.check_output(command_test, shell=True)
#最終出力ファイルのPATH(quad3_4_cionly.csv)

import math

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

outputfile = "/Users/IkkiTanaka/Documents/kaggle/Otto/sub_vw.csv"

with open(outputfile,"wb") as outfile:
    outfile.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
    for line in open("/Users/IkkiTanaka/Documents/kaggle/Otto/test.txt"):
        if line[0] == "0": #skip first 9 lines of each prediction
            continue
        #process the 10th line, which does not begin with a 0
        row = line.strip().split(" ")
        outfile.write("%s,"%row[9]) #write Id
        for m in range(8):
            outfile.write("%f,"%(zygmoid(float(row[m][2:])))) #write prediction for first 8 classes
        outfile.write("%f\n"%(zygmoid(float(row[8][2:])))) #write prediction for 9th class
