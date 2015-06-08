import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
import theano
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
import theano

import csv
import random
import numpy  as np


#load train set
def loadTrainSet():
	traindata = []
	trainlabel = []
	table = {"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
	with open("/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv") as f:
		rows = csv.reader(f)
		rows.next()
		for row in rows:
			l = []
			for i in range(1,94):
				l.append(float(row[i]))
			traindata.append(l)
			trainlabel.append(table.get(row[-1]))
	f.close()

	traindata = np.array(traindata,dtype="float")
	trainlabel = np.array(trainlabel,dtype="int")
	#Standardize(zero-mean,nomalization)
	mean = traindata.mean(axis=0)
	std = traindata.std(axis=0)
	traindata = (traindata - mean)/std
	
	#shuffle the data
	#randomIndex = [i for i in xrange(len(trainlabel))]
	#random.shuffle(randomIndex)
	#traindata = traindata[randomIndex]
	#trainlabel = trainlabel[randomIndex]
	return traindata,trainlabel

#load test set
def loadTestSet():
	testdata = []
	with open("/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv") as f:
		rows = csv.reader(f)
		rows.next()
		for row in rows:
			l = []
			for i in range(1,94):
				l.append(float(row[i]))
			testdata.append(l)
	f.close()
	testdata = np.array(testdata,dtype="float")
	#Standardize(zero-mean,nomalization)
	mean = testdata.mean(axis=0)
	std = testdata.std(axis=0)
	testdata = (testdata - mean)/std
	return testdata

def loaddata():
	print "loading data..."
	#load data in train.csv, divided into train data and validation data
	data,label = loadTrainSet()
	val_data = data[0:6000]
	val_label = label[0:6000]
	train_data = data[6000:]
	train_label = label[6000:]
        train_data_all = data
        train_label_all = label
	#load data in test.csv
	test_data = loadTestSet()
	return train_data,train_label,val_data,val_label,test_data, train_data_all, train_label_all

def shuffle(sf):
    sf['_id'] = [random.random() for i in xrange(sf.num_rows())]
    sf = sf.sort('_id')
    del sf['_id']
    return sf

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    #np.random.shuffle(X)
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


def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

def evaluation(label,pred_label):
	num = len(label)
	logloss = 0.0
	for i in range(num):
		p = max(min(pred_label[i][label[i]-1],1-10**(-15)),10**(-15))
		logloss += np.log(p)
	logloss = -1*logloss/num
	return logloss

def saveResult(testlabel,filename = "submission.csv"):
	with open(filename,'wb') as myFile:
		myWriter=csv.writer(myFile)
		myWriter.writerow(["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])
		id_num = 1
		for eachlabel in testlabel:
			l = []
			l.append(id_num)
			l.extend(eachlabel)
			myWriter.writerow(l)
			id_num += 1
train_data,train_label,val_data,val_label,test_data,train_data_all, train_label_all = loaddata()


import xgboost as xgb

#dtrain = xgb.DMatrix('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')

train_label1 = train_label - 1
dtrain = xgb.DMatrix(train_data,label=train_label1)
val_label1 = val_label - 1
dval = xgb.DMatrix(val_data,label=val_label1)
dtest = xgb.DMatrix(test_data)


randomIndex = [i for i in xrange(len(train_label_all))]
random.shuffle(randomIndex)
train_data_all_sf = train_data_all[randomIndex]
train_label_all_sf = train_label_all[randomIndex]


train_label_all_sf = train_label_all_sf - 1
dtrain_all_sf = xgb.DMatrix(train_data_all_sf,label=train_label_all_sf)
for i in xrange(0,50):

    param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(5,20),
              "bst:eta" :  round(random.uniform(.05, .25),2),#step_size
              "bst:gamma" :  random.randint(1,2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(2,8),
              "bst:subsample" :  round(random.uniform(.5, .9),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.5, .9),2),#column_subsample
              "silent": 1,
              "nthread" : 8}
    num_round = int(random.randint(250,330))
    #k = int(round(1+np.log2(dtrain_all_sf.num_row())))
    print param
    print num_round
    #print k
    #evallist  = [(dval, 'eval'),(dtrain,'train')]
    #bst = xgb.train(param,dtrain, num_round,evallist)

    watchlist  = [(dtrain_all_sf,'train')]
    #bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
    bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
    



param = {'bst:subsample': 0.87, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.56, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.64, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.18, 'colsample_bytree': 0.69, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 19, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.88, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.14, 'colsample_bytree': 0.64, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 14, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.52, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.56, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.68, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.14, 'colsample_bytree': 0.6, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 9, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.58, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.1, 'colsample_bytree': 0.66, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 10, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.89, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.09, 'colsample_bytree': 0.52, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 16, 'objective': 'multi:softprob', 'bst:min_child_weight': 7}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.83, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.73, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.83, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.73, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.56, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.08, 'colsample_bytree': 0.76, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 7}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.77, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.09, 'colsample_bytree': 0.89, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 300
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.8, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.07, 'colsample_bytree': 0.61, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 19, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 350
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.66, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.71, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 20, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 350
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.61, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.1, 'colsample_bytree': 0.65, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 350
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.67, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.6, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 13, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 350
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param

param = {'bst:subsample': 0.74, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.1, 'colsample_bytree': 0.69, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 16, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst=xgb.cv(param, dtrain_all_sf, num_round, nfold=5, seed = 407, show_stdv = False)
print param



'''
0.461124
{'bst:subsample': 0.64, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.78, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
252

0.457646
{'bst:subsample': 0.82, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.11, 'colsample_bytree': 0.67, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 18, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
265

0.457906
{'bst:subsample': 0.84, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.77, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
266

0.462017
{'bst:subsample': 0.75, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.15, 'colsample_bytree': 0.84, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 14, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
230

0.459995
{'bst:subsample': 0.72, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.67, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'multi:softprob', 'bst:min_child_weight': 8}
264

0.461664
{'bst:subsample': 0.75, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.15, 'colsample_bytree': 0.84, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 14, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
205
'''
#葉のindex予測




#0.461124
param = {'bst:subsample': 0.64, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.78, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 252
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv', index_label='id')


#0.457646
param = {'bst:subsample': 0.82, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.11, 'colsample_bytree': 0.67, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 18, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
num_round = 265
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv', index_label='id')

#0.457906
param = {'bst:subsample': 0.84, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.77, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 266
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv', index_label='id')

#0.462017
param = {'bst:subsample': 0.75, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.15, 'colsample_bytree': 0.84, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 14, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 230
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv', index_label='id')

#0.459995
param = {'bst:subsample': 0.72, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.67, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'multi:softprob', 'bst:min_child_weight': 8}
num_round = 264
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv', index_label='id')

#0.461664
param = {'bst:subsample': 0.75, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.15, 'colsample_bytree': 0.84, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 14, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 205
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv', index_label='id')

#0.461739
param = {'bst:subsample': 0.64, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.77, 'silent': 1, 'bst:gamma': 0, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
num_round = 185
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv', index_label='id')





'''
xgb_count8からtrain.csv使用
'''
#0.466713
param = {'bst:subsample': 0.73, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.2, 'colsample_bytree': 0.7, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 13, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
num_round = 182
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count8.csv', index_label='id')

#0.460912
param = {'bst:subsample': 0.74, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.14, 'colsample_bytree': 0.72, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 18, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 275
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count9.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.68, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.16, 'colsample_bytree': 0.57, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 18, 'objective': 'multi:softprob', 'bst:min_child_weight': 7}
num_round = 205
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count10.csv', index_label='id')




#0.462760
param = {'bst:subsample': 0.87, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.56, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count11.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.88, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.14, 'colsample_bytree': 0.64, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 14, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 350
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count13.csv', index_label='id')


#0.462760
param = {'bst:subsample': 0.68, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.14, 'colsample_bytree': 0.6, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 9, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count15.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.58, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.1, 'colsample_bytree': 0.66, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 10, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count16.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.89, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.09, 'colsample_bytree': 0.52, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 16, 'objective': 'multi:softprob', 'bst:min_child_weight': 7}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count17.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.83, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.73, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 6}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count19.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.56, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.08, 'colsample_bytree': 0.76, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 12, 'objective': 'multi:softprob', 'bst:min_child_weight': 7}
num_round = 450
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count20.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.77, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.09, 'colsample_bytree': 0.89, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 17, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 350
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count21.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.8, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.07, 'colsample_bytree': 0.61, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 19, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count22.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.66, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.13, 'colsample_bytree': 0.71, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 20, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 400
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count23.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.61, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.1, 'colsample_bytree': 0.65, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 15, 'objective': 'multi:softprob', 'bst:min_child_weight': 5}
num_round = 315
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count24.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.67, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.12, 'colsample_bytree': 0.6, 'silent': 1, 'bst:gamma': 1, 'nthread': 8, 'bst:max_depth': 13, 'objective': 'multi:softprob', 'bst:min_child_weight': 4}
num_round = 300
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count25.csv', index_label='id')

#0.462760
param = {'bst:subsample': 0.74, 'eval_metric': 'mlogloss', 'num_class': 9, 'bst:eta': 0.1, 'colsample_bytree': 0.69, 'silent': 1, 'bst:gamma': 2, 'nthread': 8, 'bst:max_depth': 16, 'objective': 'multi:softprob', 'bst:min_child_weight': 3}
num_round = 480
watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)
pred_prob = bst.predict( dtest )
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count26.csv', index_label='id')


