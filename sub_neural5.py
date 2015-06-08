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
X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/kaggle/Otto/train.csv')
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/kaggle/Otto/test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.2,
                 dense0_num_units=800,
                 dropout0_p=0.5,
                 dense1_num_units=500,
                 dropout1_p=0.5,
                 dense2_num_units=300,
                 dropout2_p=0.5,
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
                 random_state = 19920407,
                 eval_size=0.2,
                 batch_iterator_train=BatchIterator(batch_size=128),
                 batch_iterator_test=BatchIterator(batch_size=128 ),
                 verbose=1,
                 max_epochs=85)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_drop.csv')


layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.1,
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=900,
                 dropout1_p=0.5,
                 dense2_num_units=400,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=70)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_drop.csv')

layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.1,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=90)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_drop.csv')

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
                 dropout_in_p=0.2,
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
                 #],

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=70)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_drop.csv')


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
           ('dense4', DenseLayer),
           ('dropout4', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.2,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=90)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_drop.csv')

layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.5,
                 dense0_num_units=1000,
                 dropout0_p=0.5,
                 dense1_num_units=500,
                 dropout1_p=0.5,
                 dense2_num_units=250,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=115)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub6_drop.csv')


layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.5,
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=900,
                 dropout1_p=0.5,
                 dense2_num_units=400,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=120)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub7_drop.csv')

layers0 = [('input', InputLayer),
           ('dropout_in', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.5,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=90)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub8_drop.csv')

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
                 dropout_in_p=0.5,
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
                 #],

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=120)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub9_drop.csv')


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
           ('dense4', DenseLayer),
           ('dropout4', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout_in_p=0.5,
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

                 eval_size=0.001,
                 verbose=1,
                 max_epochs=140)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub10_drop.csv')

