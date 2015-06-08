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

import csv
import random
import numpy  as np


#load train set
def loadTrainSet():
	traindata = []
	trainlabel = []
	table = {"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
	with open("/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv") as f:
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
	randomIndex = [i for i in xrange(len(trainlabel))]
	random.shuffle(randomIndex)
	traindata = traindata[randomIndex]
	trainlabel = trainlabel[randomIndex]
	return traindata,trainlabel

#load test set
def loadTestSet():
	testdata = []
	with open("/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv") as f:
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

from sklearn.ensemble import ExtraTreesClassifier
rfClf = RandomForestClassifier(n_estimators=500,oob_score=False,bootstrap=False, max_features='sqrt',n_jobs=-1,random_state=400,criterion="gini")
rfClf.fit(train_data, train_label)
val_pred_label = rfClf.predict_proba(val_data)
print evaluation(val_label,val_pred_label)

#rfClf = RandomForestClassifier(n_estimators=400,n_jobs=-1)
#rfClf.fit(X, y)
#evaluate on validation set
val_pred_label = rfClf.predict_proba(test_data)
print val_pred_label




rfClf = RandomForestClassifier(n_estimators=400,bootstrap=False,n_jobs=-1,random_state=400,class_weight="auto")
rfClf.fit(train_data_all, train_label_all)
val_pred_label = rfClf.predict_proba(test_data)

saveResult(val_pred_label,filename ='/Users/IkkiTanaka/Documents/kaggle/Otto/rf_submission6.csv')

zz = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/rf_submission6.csv')


rf_submission1  0.529432526824
RandomForestClassifier(n_estimators=800,n_jobs=-1)
rf_submission2  0.528569368184
RandomForestClassifier(n_estimators=800,max_features='sqrt',n_jobs=-1,random_state=400)
rf_submission3  0.527540024873
(n_estimators=400, max_features='sqrt',n_jobs=-1,random_state=400)
rf_submission4  0.500978828727
RandomForestClassifier(n_estimators=400,bootstrap=False, max_features='sqrt',n_jobs=-1,random_state=400)
rf_submission5  0.501592056555
RandomForestClassifier(n_estimators=500,bootstrap=False, max_features='sqrt',n_jobs=-1,random_state=400)
rf_submission6  
RandomForestClassifier(n_estimators=400,bootstrap=False, max_features='sqrt',n_jobs=-1,random_state=400,class_weight='auto')



adClf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=550, learning_rate=0.1, )
adClf.fit(train_data, train_label)
val_pred_label = adClf.predict_proba(val_data)
print evaluation(val_label,val_pred_label)

#rfClf = RandomForestClassifier(n_estimators=400,n_jobs=-1)
#rfClf.fit(X, y)
#evaluate on validation set
test_label = adClf.predict_proba(test_data)
print test_label
saveResult(test_label,filename ='/Users/IkkiTanaka/Documents/kaggle/Otto/ad_submission1.csv')




"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Abhishek Thakur
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn import utils

# import data
train = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')
test = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd.csv')
sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

train, labels = utils.shuffle(train,labels, random_state=400)
tr, va, la_tr, la_va = train[1000:],train[:1000],labels[1000:],labels[:1000]
# train a random forest classifier
clf = RandomForestClassifier(n_estimators=400,oob_score=False,bootstrap=False, max_features='sqrt',n_jobs=4,random_state=400,criterion="gini")
clf.fit(tr, la_tr)
preds_va = clf.predict_proba(va)
print evaluation((la_va+1),(preds_va+0.000001))

la_va = la_va+0.001



clf = RandomForestClassifier(n_estimators=800,oob_score=False,bootstrap=False, max_features='sqrt',n_jobs=-1,random_state=400,criterion="gini")
clf.fit(train, labels)



# predict on test set
preds = clf.predict_proba(test)

# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/rf_2.csv', index_label='id')



import xgboost as xgb

#dtrain = xgb.DMatrix('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd.csv')

train_label1 = train_label - 1
dtrain = xgb.DMatrix(train_data,label=train_label1)
val_label1 = val_label - 1
dval = xgb.DMatrix(val_data,label=val_label1)
dtest = xgb.DMatrix(test_data)
train_label_all1 = train_label_all - 1
dtrain_all = xgb.DMatrix(train_data_all,label=train_label_all1)

param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  8,
              "bst:eta" :  0.22,
              "bst:gamma" :  1,
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.80,
              "colsample_bytree" :  0.70,
              "nthread" : 8,
              "silent": 1
              }
num_round = 250

evallist  = [(dval, 'eval'),(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,evallist)

watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb3.csv', index_label='id')



'''
xgb1
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  10,
              "bst:eta" :  0.2,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.9,#row_subsample
              "colsample_bytree" :  0.8,#column_subsample
              "nthread" : 8}
num_round = 250

xgb2
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  11,
              "bst:eta" :  0.25,
              "bst:gamma" :  1.25,
              "bst:min_child_weight" :  4.2,
              "bst:subsample" :  0.85,
              "colsample_bytree" :  0.75,
              "nthread" : 8,
              "silent": 1
              }

xgb3
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  8,
              "bst:eta" :  0.22,
              "bst:gamma" :  1,
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.80,
              "colsample_bytree" :  0.70,
              "nthread" : 8,
              "silent": 1
              }
num_round = 250

'''

'''
submission(graphLab)のをxgbで実行
'''
#submission->xgb_sub
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  10,
              "bst:eta" :  0.3,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.9,#row_subsample
              "colsample_bytree" :  0.8,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 250
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub.csv', index_label='id')


#submission1->xgb_sub1
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  8,
              "bst:eta" :  0.3,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.9,#row_subsample
              "colsample_bytree" :  0.8,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 250
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub1.csv', index_label='id')


#submission2->xgb_sub2
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  6,
              "bst:eta" :  0.3,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.9,#row_subsample
              "colsample_bytree" :  0.9,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 250
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub2.csv', index_label='id')


#submission5->xgb_sub5
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  10,
              "bst:eta" :  0.3,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.6,#row_subsample
              "colsample_bytree" :  0.75,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 250
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub5.csv', index_label='id')

#submission6->xgb_sub6
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  8,
              "bst:eta" :  0.3,#step_size
              "bst:gamma" :  0.8,#min_loss_reduction
              "bst:min_child_weight" :  3,
              "bst:subsample" :  0.7,#row_subsample
              "colsample_bytree" :  0.8,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 300
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub6.csv', index_label='id')

#submission7->xgb_sub7
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  8,
              "bst:eta" :  0.2,#step_size
              "bst:gamma" :  0.8,#min_loss_reduction
              "bst:min_child_weight" :  3,
              "bst:subsample" :  0.7,#row_subsample
              "colsample_bytree" :  0.8,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 300
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub7.csv', index_label='id')

#submission8->xgb_sub8
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  10,
              "bst:eta" :  0.2,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.9,#row_subsample
              "colsample_bytree" :  0.8,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 250
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub8.csv', index_label='id')

#submission9->xgb_sub9
param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  10,
              "bst:eta" :  0.1,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.9,#row_subsample
              "colsample_bytree" :  0.9,#column_subsample
              "nthread" : 8,
              "silent": 1
              }
num_round = 300
watchlist  = [(dtrain_all,'train')]
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred_prob = bst.predict( dtest )

sample = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sampleSubmission.csv')
preds = pd.DataFrame(pred_prob, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_sub9.csv', index_label='id')



