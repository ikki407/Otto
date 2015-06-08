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
	with open("/Users/IkkiTanaka/Documents/kaggle/Otto/train_count.csv") as f:
		rows = csv.reader(f)
		rows.next()
		for row in rows:
			l = []
			for i in range(1,280):
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
	with open("/Users/IkkiTanaka/Documents/kaggle/Otto/test_count.csv") as f:
		rows = csv.reader(f)
		rows.next()
		for row in rows:
			l = []
			for i in range(1,280):
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

param = {"objective" : "multi:softprob",
              "eval_metric" :  "mlogloss",
              "num_class" :  9,
              "nthread" :  8,
              "bst:max_depth" :  18,
              "bst:eta" :  0.2,#step_size
              "bst:gamma" :  1,#min_loss_reduction
              "bst:min_child_weight" :  4,
              "bst:subsample" :  0.7,#row_subsample
              "colsample_bytree" :  0.6,#column_subsample
              "silent": 1,
              "nthread" : 8}
num_round = 150

#evallist  = [(dval, 'eval'),(dtrain,'train')]
#bst = xgb.train(param,dtrain, num_round,evallist)

watchlist  = [(dtrain_all_sf,'train')]
bst = xgb.train(param,dtrain_all_sf, num_round,watchlist)

#葉のindex予測
print train_label_all
train_label_all = train_label_all - 1

dtrain_all = xgb.DMatrix(train_data_all)

pred_leaf_tr = bst.predict( dtrain_all ,pred_leaf = True)
pred_leaf_te = bst.predict( dtest ,pred_leaf = True)



mean_leaf = np.mean(pred_leaf_tr,axis=0)
std_leaf = np.std(pred_leaf_tr,axis=0)
pred_leaf_tr = (pred_leaf_tr - mean_leaf)/std_leaf

index_tr = np.where(~np.isnan(pred_leaf_tr[0]))

mean_leaf = np.mean(pred_leaf_te,axis=0)
std_leaf = np.std(pred_leaf_te,axis=0)
pred_leaf_te = (pred_leaf_te - mean_leaf)/std_leaf

index_te = np.where(~np.isnan(pred_leaf_te[0]))

if not (index_tr[0] == index_te[0]):
    if len(index_tr[0])<len(index_te[0]):
        index = index_tr[0]
    else:
        index = index_te[0]


pred_leaf_tr = pred_leaf_tr[:,index]
#pred_leaf_tr_drop = DataFrame(pred_leaf_tr).dropna(1)
#pred_leaf_tr_drop = pred_leaf_tr_drop.values.tolist()
pred_leaf_tr = pred_leaf_tr.reshape(pred_leaf_tr.shape[0],pred_leaf_tr.shape[1])

pred_leaf_te = pred_leaf_te[:,index]
pred_leaf_te = pred_leaf_te.reshape(pred_leaf_te.shape[0],pred_leaf_te.shape[1])

#pred_leaf_te_drop = DataFrame(pred_leaf_te).dropna(1)
#pred_leaf_te_drop = pred_leaf_te_drop.values.tolist()

X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count.csv')
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count.csv', scaler)
num_classes = len(encoder.classes_)

new_X = np.c_[X,pred_leaf_tr]
new_X_test = np.c_[X_test,pred_leaf_te]
#randomIndex = [i for i in xrange(len(trainlabel))]
#random.shuffle(randomIndex)
print y

randomIndex = [i for i in xrange(len(y))]
random.shuffle(randomIndex)
new_X = new_X[randomIndex]
y = y[randomIndex]

num_features = new_X.shape[1]
print new_X_test.shape
print new_X.shape

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=1000,
                 dropout1_p=0.5,
                 dense2_num_units=600,
                 dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.005,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.005)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=15)
net0.fit(new_X, y)

make_submission(net0, new_X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN2.csv')

a=pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN2.csv')
a['id'] = a['id'].astype(int)
a.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/Submission4.csv',index=False)



layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=900,
                 dropout1_p=0.5,
                 dense2_num_units=600,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.005,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.005)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=20)
net0.fit(new_X, y)
make_submission(net0, new_X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN3.csv')

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('dense3', DenseLayer),
           ('dropout3', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 dense2_num_units=750,
                 dropout2_p=0.4,
                 dense3_num_units=400,
                 dropout3_p=0.3,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.005)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #]2

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=20)
net0.fit(new_X, y)
make_submission(net0, new_X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN4.csv')


layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('dense3', DenseLayer),
           ('dropout3', DropoutLayer),
           ('dense4', DenseLayer),
           ('dropout4', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=650,
                 dropout0_p=0.5,
                 dense1_num_units=1350,
                 dropout1_p=0.5,
                 dense2_num_units=1550,
                 dropout2_p=0.5,
                 dense3_num_units=900,
                 dropout3_p=0.5,
                 dense4_num_units=350,
                 dropout4_p=0.3,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.005,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.005)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=20)
net0.fit(new_X, y)
make_submission(net0, new_X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN5.csv')


sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN2.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN3.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN4.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN5.csv')

#submission1と2を使ってみる. 3は化学臭すぎ?


final_sub = (sub1+sub2+sub3+sub4)/4.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN2_4.csv',index = False)




