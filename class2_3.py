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
import lasagne.nonlinearities as nonlin
from lasagne.updates import adagrad, adadelta
import random
from nolearn.lasagne import BatchIterator

import csv
import random
import numpy  as np




#############################################################
#
#モデルを作る前にtrainデータとtestデータでのclass2 or 3であるデータを抽出する
#
#############################################################

#train data
train = pd.read_csv("/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv")
index23 = [any(x) for x in pd.concat([train['target']=='Class_2',train['target']=='Class_3'],axis=1).values.tolist()]
train23 = train.ix[index23]
#predictionからclass2or3で迷っているindexを抽出する

pred = pd.read_csv("/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop20.csv")
test = pd.read_csv("/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv")

index = []
for i in xrange(0,len(pred)):
    rank23 = pred.iloc[i,1:].argsort()[-2:].values.tolist()
    if rank23 == [1,2] or rank23 == [2,1]:
        #print rank23
        #print '%iはClass2,3が上位'
        kijun = pred.iloc[i,2] > 0.4 and pred.iloc[i,3] > 0.4
        print pred.iloc[i,2] + pred.iloc[i,3]
        if kijun:
            print '%iを追加' % i
            index.append(i)

test23 = test.iloc[index,:]
pred23_old = pred.iloc[index,2:4]
train23.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_23only.csv',index=False)
test23.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_23only.csv',index=False)
#予測したら
#pred.iloc[index,2:4] = pred23

################################################################
#2,3だけのデータ
#train23
#test23
################################################################
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
    test_index = df.index    
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids, test_index

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

X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count_23only.csv')
X_test, ids, test_index = load_test_data('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count_23only.csv', scaler)
num_classes = len(encoder.classes_)
print len(X)
print len(X_test)

#X_rand = np.random.normal(size=X.shape)*0.01
#X = X + X_rand

X = X[:,[13,24,32,39,42,47,71]]
X_test = X_test[:,[13,24,32,39,42,47,71]]

num_features = X.shape[1]

for i in xrange(0,50):
    layers0 = [('input', InputLayer),
            ('dropout_in', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               #('dense1', DenseLayer),
               #('dropout1', DropoutLayer),
               #('dense2', DenseLayer),
               #('dropout2', DropoutLayer),
               #('dense3', DenseLayer),
               #('dropout3', DropoutLayer),
               #('dense4', DenseLayer),
               #('dropout4', DropoutLayer),
               ('output', DenseLayer)]
    dense_list = [random.randint(30, 100),random.randint(70,150),random.randint(50, 150),random.randint(100, 350)]
    leakness_list = [random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005)]
    max_iter = random.randint(100,170)
    #dropout_in = round(random.uniform(.1, .3),2)
    dropout_in = 0.0
    print dense_list
    print leakness_list
    print max_iter
    print dropout_in
        
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.0,
                     #dense0_nonlinearity=sigmoid,
                     #dense1_num_units=dense_list[1],
                     #dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     #dense2_num_units=dense_list[2],
                     #dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=dense_list[3],
                     #dropout3_p=0.5,
                     #dense4_num_units=dense_list[4],
                     #dropout4_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    #dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    #dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    #dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
                    #dense4_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[4]),
                    #dense0_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense1_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense2_nonlinearity= nonlin.LeakyRectify(0.1),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    update=adagrad,
                    #update=adadelta,
                    #update=nesterov_momentum,

                    update_learning_rate=0.01,
                    #update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    #on_epoch_finished=[
                    #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    #],
                    random_state = 407,
                    #l2_costs=0.0001,
                    eval_size=0.2,
                    verbose=1,
                    batch_iterator_train=BatchIterator(batch_size=30),
                    batch_iterator_test=BatchIterator(batch_size=30 ),

                    max_epochs=1000)
    net0.fit(X, y)
    val_loss = []
    for j in xrange(0,max_iter):
        val_loss.append(net0.train_history_[j]['valid_loss'])
    if min(val_loss) < 0.48:
        max_iter2 = pd.Series(val_loss).argsort()[0] + 15

        net0 = NeuralNet(layers=layers0,                     
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     #dense2_num_units=dense_list[2],
                     #dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=dense_list[3],
                     #dropout3_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    #dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    #dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
                    #dense0_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense1_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense2_nonlinearity= nonlin.LeakyRectify(0.1),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    update=adagrad,
                    #update=adadelta,
                    #update=nesterov_momentum,

                    update_learning_rate=0.01,
                    #update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    #on_epoch_finished=[
                    #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    #],
                    random_state = 407,
                    #l2_costs=0.0001,
                    eval_size=0.001,
                    verbose=1,
                    batch_iterator_train=BatchIterator(batch_size=128),
                    batch_iterator_test=BatchIterator(batch_size=128 ),

                    max_epochs=max_iter2)
        net0.fit(X, y)
        for j in xrange(0,max_iter2):
            val_loss.append(net0.train_history_[j]['valid_loss'])
        min_loss = min(val_loss)

        submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/class23lrelu4ada%i_%f_d%i_%i_%i_%i_l%f_%f_%f_%f_dr%f_m%i.csv' % (i,min_loss,dense_list[0],dense_list[1],dense_list[2],dense_list[3],leakness_list[0],leakness_list[1],leakness_list[2],leakness_list[3],dropout_in,max_iter2)
        make_submission(net0, X_test, ids, encoder, name=submission_name)

























layers0 = [('input', InputLayer),
            ('dropout_in', DropoutLayer),
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
                 dropout_in_p=0.3,
                 dense0_num_units=1000,
                 dropout0_p=0.5,
                 dense1_num_units=700,
                 dropout1_p=0.5,
                 dense2_num_units=500,
                 dropout2_p=0.5,
                 dense3_num_units=250,
                 dropout3_p=0.5,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 dense3_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 
                 #update=nesterov_momentum,
                 update_learning_rate=0.01,
                 #update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.005)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],
                 batch_iterator_train=BatchIterator(batch_size=128),
                 batch_iterator_test=BatchIterator(batch_size=128),

                 eval_size=0.2,
                 verbose=1,
                 max_epochs=65)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/NN1_class23only.csv')

layers0 = [('input', InputLayer),
            ('dropout_in', DropoutLayer),
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
                 dropout_in_p=0.3,
                 dense0_num_units=1000,
                 dropout0_p=0.5,
                 dense1_num_units=700,
                 dropout1_p=0.5,
                 dense2_num_units=500,
                 dropout2_p=0.5,
                 dense3_num_units=250,
                 dropout3_p=0.5,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 dense3_nonlinearity= nonlin.LeakyRectify(leakiness=0.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 
                 #update=nesterov_momentum,
                 update_learning_rate=0.01,
                 #update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.005)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],
                 batch_iterator_train=BatchIterator(batch_size=68),
                 batch_iterator_test=BatchIterator(batch_size=68),

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=65)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/NN2_class23only.csv')

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           #('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=550,
                 dropout0_p=0.5,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense2_num_units=250,
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
                 #],

                 eval_size=0.2,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/NN3_class23only.csv')



















pred23_1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/NN1_class23only.csv')
#pred23_2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/NN2_class23only.csv')
#pred23_3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/NN3_class23only.csv')

pred23 = pred23_1#(pred23_1+pred23_2+pred23_3)/3.0
pred23.index = pred.iloc[index,2:4].index
pred.iloc[index,2:4] = pred23.iloc[:,1:]

pred.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/NN1_class23only_final_sub2.csv',index=False)

NN1_class23only_final_sub1
0.41978
