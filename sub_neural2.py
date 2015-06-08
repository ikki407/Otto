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

X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/kaggle/Otto/train_count.csv')
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/kaggle/Otto/test_count.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=150,
                 dropout0_p=0.5,
                 dense1_num_units=300,
                 dropout1_p=0.5,
                 dense2_num_units=500,
                 dropout2_p=0.5,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 #update_learning_rate=0.01,
                 #update_momentum=0.9,
                 
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
    
                 on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 ],
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=50)

net0.fit(X, y)
yp = DataFrame(net0.predict_proba(X_test),columns=[ u'Class_1', u'Class_2', u'Class_3', u'Class_4', u'Class_5', u'Class_6', u'Class_7', u'Class_8', u'Class_9'])


make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission2.csv')


sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission1.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission2.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission3.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission1.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission3.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission4.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission5.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission6.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission7.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission8.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/rf_submission1.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub17)/13
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8_sub9_neural_1_3_4.csv',index = False)

'''
sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8neural_net_submission1
0.42830

sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8_sub9_neural_1_3_4
0.42516

sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8_sub9_neural_1_3_4_6
0.42606

sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8_sub9_neural_1_3_4_8
0.42697

sub_sub1_sub2_sub3_sub5_sub6_sub7_sub8_sub9_neural_1_3_4_rf1
0.42771

sub_sub5_sub6_sub7_sub8_sub9_neural_1_3_4
0.42469

sub7_sub8_sub9_neural_1_3_4.csv.zip
0.42978

sub_sub5_sub6_sub7_sub8_sub9_sub10_sub11_neural_1_3
0.42524

sub_sub5_sub6_sub7_sub8_sub9_plus_log_neural_1_3
0.42485


sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission1.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission3.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission4.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_log.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission1_log.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission2_log.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission3_log.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_log.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_log.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_log.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16)/16
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sub_sub5_sub6_sub7_sub8_sub9_plus_log_neural_1_3.csv',index = False)

'''

X, y, encoder, scaler = load_train_data('/Users/IkkiTanaka/Documents/kaggle/Otto/train_log.csv')
X_test, ids = load_test_data('/Users/IkkiTanaka/Documents/kaggle/Otto/test_log.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]


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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=50)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission10.csv')


0.483190   LB 0.48011 neural_net_submission1.csv   45epoch
  DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 250)         	produces     250 outputs
  DropoutLayer      	(None, 500)         	produces     500 outputs
  DenseLayer        	(None, 500)         	produces     500 outputs
  DropoutLayer      	(None, 1000)        	produces    1000 outputs
  DenseLayer        	(None, 1000)        	produces    1000 outputs
  InputLayer        	(None, 93)          	produces      93 outputs


0.483    LB         neural_net_submission2.csv
 DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 150)         	produces     150 outputs
  DropoutLayer      	(None, 300)         	produces     300 outputs
  DenseLayer        	(None, 300)         	produces     300 outputs
  DropoutLayer      	(None, 500)         	produces     500 outputs
  DenseLayer        	(None, 500)         	produces     500 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

0.481641                   neural_net_submission3.csv 50epoch
  DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 400)         	produces     400 outputs
  DropoutLayer      	(None, 900)         	produces     900 outputs
  DenseLayer        	(None, 900)         	produces     900 outputs
  DropoutLayer      	(None, 1350)        	produces    1350 outputs
  DenseLayer        	(None, 1350)        	produces    1350 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

0.478916                   neural_net_submission4.csv 70epoch lr=0.005
  DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 600)         	produces     600 outputs
  DropoutLayer      	(None, 900)         	produces     900 outputs
  DenseLayer        	(None, 900)         	produces     900 outputs
  DropoutLayer      	(None, 1350)        	produces    1350 outputs
  DenseLayer        	(None, 1350)        	produces    1350 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

0.480981                    neural_net_submission5.csv 70epoch lr=0.005
  DenseLayer        	(None, 9)           	produces       9 outputs
  DropoutLayer      	(None, 750)         	produces     750 outputs
  DenseLayer        	(None, 750)         	produces     750 outputs
  DropoutLayer      	(None, 1250)        	produces    1250 outputs
  DenseLayer        	(None, 1250)        	produces    1250 outputs
  InputLayer        	(None, 93)          	produces      93 outputs


0.482553                    neural_net_submission6.csv 70epoch lr=0.005
  DenseLayer        	(None, 9)           	produces       9 outputs
  DropoutLayer      	(None, 1250)        	produces    1250 outputs
  DenseLayer        	(None, 1250)        	produces    1250 outputs
  DropoutLayer      	(None, 750)         	produces     750 outputs
  DenseLayer        	(None, 750)         	produces     750 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

0.477834                   neural_net_submission7.csv 70epoch 
  DenseLayer        	(None, 9)           	produces       9 outputs
  DropoutLayer      	(None, 1250)        	produces    1250 outputs
  DenseLayer        	(None, 1250)        	produces    1250 outputs
  DropoutLayer      	(None, 1050)        	produces    1050 outputs
  DenseLayer        	(None, 1050)        	produces    1050 outputs
  InputLayer        	(None, 93)          	produces      93 outputs
on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 ],

0.477834                   neural_net_submission8.csv 70epoch 
  DenseLayer        	(None, 9)           	produces       9 outputs
  DropoutLayer      	(None, 1500)        	produces    1500 outputs
  DenseLayer        	(None, 1500)        	produces    1500 outputs
  DropoutLayer      	(None, 1250)        	produces    1250 outputs
  DenseLayer        	(None, 1250)        	produces    1250 outputs
  InputLayer        	(None, 93)          	produces      93 outputs
on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 ],

                   neural_net_submission10.csv 50epoch
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 dense2_num_units=750,
                 dropout2_p=0.4,
                 dense3_num_units=400,
                 dropout3_p=0.3,



'''
sub_neural_all

   neural_net_sub1_all.csv   45epoch
  DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 250)         	produces     250 outputs
  DropoutLayer      	(None, 500)         	produces     500 outputs
  DenseLayer        	(None, 500)         	produces     500 outputs
  DropoutLayer      	(None, 1000)        	produces    1000 outputs
  DenseLayer        	(None, 1000)        	produces    1000 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

                 neural_net_sub2_all.csv 50epoch
  DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 400)         	produces     400 outputs
  DropoutLayer      	(None, 900)         	produces     900 outputs
  DenseLayer        	(None, 900)         	produces     900 outputs
  DropoutLayer      	(None, 1350)        	produces    1350 outputs
  DenseLayer        	(None, 1350)        	produces    1350 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

               neural_net_sub3_all.csv 70epoch lr=0.005
  DenseLayer        	(None, 9)           	produces       9 outputs
  DenseLayer        	(None, 600)         	produces     600 outputs
  DropoutLayer      	(None, 900)         	produces     900 outputs
  DenseLayer        	(None, 900)         	produces     900 outputs
  DropoutLayer      	(None, 1350)        	produces    1350 outputs
  DenseLayer        	(None, 1350)        	produces    1350 outputs
  InputLayer        	(None, 93)          	produces      93 outputs

               neural_net_sub4_all.csv 50epoch lr=0.01
                 dense0_num_units=1350,
                 dropout0_p=0.5,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 dense2_num_units=750,
                 dropout2_p=0.4,
                 dense3_num_units=400,
                 dropout3_p=0.3,

               neural_net_sub5_all.csv 70epoch lr=0.005
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 dense1_num_units=1350,
                 dropout1_p=0.5,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 dense3_num_units=600,
                 dropout3_p=0.4,
                 dense4_num_units=350,
                 dropout4_p=0.3,
              
              neural_net_sub6_all.csv 70epoch lr=0.005
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 dense1_num_units=1350,
                 dropout1_p=0.5,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 dense3_num_units=700,
                 dropout3_p=0.4,
                 dense4_num_units=550,
                 dropout4_p=0.4,

              neural_net_sub7_all.csv 70epoch lr=0.005
                 dense0_num_units=450,
                 dropout0_p=0.4,
                 dense1_num_units=950,
                 dropout1_p=0.4,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 dense3_num_units=1550,
                 dropout3_p=0.5,

              neural_net_sub8_all.csv 80epoch lr=0.005
                 dense0_num_units=650,
                 dropout0_p=0.5,
                 dense1_num_units=1350,
                 dropout1_p=0.5,
                 dense2_num_units=1650,
                 dropout2_p=0.5,
                 dense3_num_units=2000,
                 dropout3_p=0.5,

'''
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]
from lasagne.nonlinearities import sigmoid
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=1248,
                 dropout0_p=0.5,
                 dense0_nonlinearity=sigmoid,
                 dense1_num_units=950,
                 dropout1_p=0.5,
                 dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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
                 max_epochs=60)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub6_all_count.csv')




sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1_3_4
0.42292

sd_sub5_sub6_sub7_sub8_sub9_neural_all_1_3_4
0.4238...
sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__4
0.42143
sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__4_rf4
0.42290

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__6
0.42139

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5
0.42124

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_7
0.42188

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_rf1
0.42529

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_8_xgb1
0.42145

sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_xgb1<-
0.42109

sd_sub_sub5_sub8_sub9_neural_all_1__5_xgb1
0.42261

sd_xgbsub0_12_5_9_neural_all_1__5_xgb1
0.42284


sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub8_all.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission1_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission2_sd.csv')

#submission1と2を使ってみる. 3は化学臭すぎ?


final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub13)/12.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sd_xgbsub0_12_5_9_neural_all_1__5_xgb1.csv',index = False)

sub = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/rf_1.csv')


layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=55)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count.csv')

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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=60)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')

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
                 max_epochs=80)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')

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
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=60)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')


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
                 max_epochs=80)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]
from lasagne.nonlinearities import sigmoid
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=1248,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=950,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub6_all_count.csv')

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
                 dense0_num_units=1500,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=550,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub7_all_count.csv')

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
                 dense0_num_units=500,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1350,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=600,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub8_all_count.csv')

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
                 dense0_num_units=600,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1600,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub9_all_count.csv')

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           #('dense2', DenseLayer),
           #('dropout2', DropoutLayer),
           ('output', DenseLayer)]
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=1600,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=950,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 #dense2_num_units=1600,
                 #dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub10_all_count.csv')

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
                 dense0_num_units=1800,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1550,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub11_all_count.csv')

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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=750,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub12_all_count.csv')

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
                 dense0_num_units=750,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1650,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub13_all_count.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1750,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub14_all_count.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub15_all_count.csv')









































'''

import lasagne.nonlinearities as nonlin
from lasagne.updates import adagrad, adadelta

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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=850,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')

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
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')

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
                 dense0_num_units=1150,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

'''


'''
こっから本気モード
'''
import lasagne.nonlinearities as nonlin
from lasagne.updates import adagrad, adadelta
import random
from nolearn.lasagne import BatchIterator

for i in xrange(0,50):
    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense2', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]
    dense_list = [random.randint(1000, 1500),random.randint(700, 1000),random.randint(300, 700)]
    leakness_list = [random.uniform(.0001, .003),random.uniform(.0001, .003),random.uniform(.0001, .003)]
    max_iter = random.randint(30,50)
    dropout_3 = random.uniform(.3, .5)
    print dense_list
    print leakness_list
    print max_iter
    print dropout_3

    net0 = NeuralNet(layers=layers0,
                     
                     input_shape=(None, num_features),
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=dropout_3,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=900,
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
                    dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    #dense0_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense1_nonlinearity= nonlin.LeakyRectify(0.1),
                    #dense2_nonlinearity= nonlin.LeakyRectify(0.1),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    #update=adagrad,
                    update=nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,
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
                    batch_iterator_train=BatchIterator(batch_size=200),
                    batch_iterator_test=BatchIterator(batch_size=200 ),

                    max_epochs=max_iter)
    net0.fit(X, y)
    val_loss = []
    for j in xrange(0,max_iter):
        val_loss.append(net0.train_history_[j]['valid_loss']
    if min(val_loss) < 0.48:
        max_iter2 = pd.Series(val_loss).argsort()[0] + 5

        net0 = NeuralNet(layers=layers0,
                     
                     input_shape=(None, num_features),
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=dropout_3,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=900,
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
                    dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    #update=adagrad,
                    update=nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    #on_epoch_finished=[
                    #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    #],

                    eval_size=0.001,
                    verbose=1,
                    max_epochs=max_iter2)
        net0.fit(X, y)
        submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu%i_d%i_%i_%i_l%f_%f_%f_dr%f_m%i.csv' % (i,dense_list[0],dense_list[1],dense_list[2],leakness_list[0],leakness_list[1],leakness_list[2],dropout_3,max_iter)
        make_submission(net0, X_test, ids, encoder, name=submission_name)



for i in xrange(0,50):
    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense2', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]
    dense_list = [random.randint(1000, 1500),random.randint(700, 900),random.randint(300, 600)]
    #leakness_list = [random.uniform(.0001, .003),random.uniform(.0001, .003),random.uniform(.0001, .003)]
    max_iter = random.randint(30,50)
    dropout_3 = random.uniform(.3, .5)
    print dense_list
    #print leakness_list
    print max_iter
    print dropout_3
    net0 = NeuralNet(layers=layers0,
                     
                     input_shape=(None, num_features),
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=dropout_3,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=900,
                     #dropout3_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    #dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    #dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    #dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    #update=adagrad,
                    update=nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    #on_epoch_finished=[
                    #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    #],

                    eval_size=0.2,
                    verbose=1,
                    max_epochs=max_iter)
    net0.fit(X, y)
    val_loss = []
    for j in xrange(0,max_iter):
        val_loss.append(net0.train_history_[j]['valid_loss']
    if min(val_loss) < 0.47:
        max_iter2 = pd.Series(val_loss).argsort()[0] + 5

        net0 = NeuralNet(layers=layers0,
                     
                     input_shape=(None, num_features),
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=dropout_3,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=900,
                     #dropout3_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    #dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    #dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    #dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    #update=adagrad,
                    update=nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,
                    #update_learning_rate=theano.shared(float32(0.01)),
                    #update_momentum=theano.shared(float32(0.9)),
    
                    #on_epoch_finished=[
                    #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                    #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    #],

                    eval_size=0.001,
                    verbose=1,
                    max_epochs=max_iter2)
        net0.fit(X, y)

        submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/relu%i_d%i_%i_%i_dr%f_m%i.csv' % (i,dense_list[0],dense_list[1],dense_list[2],dropout_3,max_iter)
        make_submission(net0, X_test, ids, encoder, name=submission_name)



lrelu49_d1214_863_637_l0.002350_0.000907_0.002721_dr0.401939_m37.csv
0.46144


    net0 = NeuralNet(layers=layers0,
                     
                     input_shape=(None, num_features),
                     dense0_num_units=600,
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=900,
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=1200,
                     dropout2_p=dropout_3,
                     #dense2_nonlinearity=sigmoid,
                     #dense3_num_units=900,
                     #dropout3_p=0.5,
                     #dense4_num_units=900,
                     #dropout4_p=0.5,
                     #dropout2_p=0.5,
                     #dense3_num_units=550,
                     #dropout3_p=0.4,
                    #dense4_num_units=512,
                    #dropout4_p=0.3,
                    #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                    #dense0_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[0]),
                    #dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    #dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    dense0_nonlinearity= nonlin.LeakyRectify(0.01),
                    dense1_nonlinearity= nonlin.LeakyRectify(0.01),
                    dense2_nonlinearity= nonlin.LeakyRectify(0.01),
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,
                    #update=adagrad,
                    update=nesterov_momentum,
                    update_learning_rate=0.05,
                    update_momentum=0.9,
                    learn_rate_decays=0.00001,
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
                    batch_iterator_train=BatchIterator(batch_size=128),
                    batch_iterator_test=BatchIterator(batch_size=128 ),

                    max_epochs=max_iter)
    net0.fit(X, y)



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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1055,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.4,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0004),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0007),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/2.csv')

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
                 dense0_num_units=1250,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1010,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/3.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0005),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/4.csv')

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
                 dense0_num_units=1250,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=950,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.4,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0005),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0005),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0005),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/5.csv')

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
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/6.csv')

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
                 dense0_num_units=1750,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0006),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0006),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0006),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/7.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/8.csv')

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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=850,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/9.csv')

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
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/10.csv')

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
                 dense0_num_units=1150,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/11.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/12.csv')

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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=850,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/13.csv')

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
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/14.csv')

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
                 dense0_num_units=1150,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/15.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/16.csv')

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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=850,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/17.csv')

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
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/18.csv')

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
                 dense0_num_units=1150,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/19.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/20.csv')

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
                 dense0_num_units=1450,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1050,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=850,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/21.csv')

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
                 dense0_num_units=1550,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=650,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/22.csv')

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
                 dense0_num_units=1150,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1250,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=950,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/23.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/24.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1450,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1250,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
                 #dropout2_p=0.5,
                 #dense3_num_units=550,
                 #dropout3_p=0.4,
                 #dense4_num_units=512,
                 #dropout4_p=0.3,
                 #hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
                 dense0_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense1_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 dense2_nonlinearity= nonlin.LeakyRectify(leakiness=.0001),
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #update=adagrad,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 #update_learning_rate=theano.shared(float32(0.01)),
                 #update_momentum=theano.shared(float32(0.9)),
    
                 #on_epoch_finished=[
                 #   AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
                 #   AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #],

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/25.csv')


'''
Leaky Relu25個↑
普通の25個↓
'''
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/26.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/27.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/28.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/29.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/30.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/31.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/32.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/33.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/34.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/35.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/36.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/37.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/38.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/39.csv')

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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/40.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/41.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/42.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/43.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/44.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/45.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/46.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/47.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/48.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/49.csv')
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
                 dense0_num_units=850,
                 dropout0_p=0.5,
                 #dense0_nonlinearity=sigmoid,
                 dense1_num_units=1150,
                 dropout1_p=0.5,
                 #dense1_nonlinearity=sigmoid,
                 dense2_num_units=1350,
                 dropout2_p=0.5,
                 #dense2_nonlinearity=sigmoid,
                 #dense3_num_units=900,
                 #dropout3_p=0.5,
                 #dense4_num_units=900,
                 #dropout4_p=0.5,
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

                 eval_size=0.01,
                 verbose=1,
                 max_epochs=45)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/50.csv')



sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu0_d1074_707_550_dr0.464376_m39.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu0_d1402_958_570_l0.000486_0.001839_0.001046_dr0.401953_m41.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu1_d1149_926_454_dr0.439901_m37.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu1_d1338_907_605_l0.001606_0.002141_0.001350_dr0.460619_m43.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu2_d1067_921_507_dr0.436505_m42.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu3_d1269_947_504_dr0.420161_m46.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu2_d1137_952_553_l0.000216_0.002471_0.000362_dr0.374670_m48.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu4_d1196_768_328_dr0.400937_m34.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3_d1177_827_671_l0.000820_0.002572_0.002362_dr0.491182_m47.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu5_d1392_977_668_dr0.432721_m32.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4_d1486_816_413_l0.001423_0.001336_0.002512_dr0.356864_m39.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu6_d1096_923_436_dr0.460941_m42.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu7_d1275_742_309_dr0.480392_m32.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu5_d1038_780_611_l0.000547_0.002009_0.000419_dr0.319871_m44.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu8_d1193_936_459_dr0.344566_m46.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu6_d1232_941_515_l0.000155_0.002482_0.001930_dr0.399731_m41.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu9_d1260_874_516_dr0.331898_m46.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu7_d1024_885_564_l0.000292_0.002399_0.001296_dr0.308141_m39.csv')
sub20 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu10_d1086_744_463_dr0.327222_m48.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu8_d1011_826_659_l0.000589_0.000173_0.001879_dr0.431526_m36.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu9_d1420_839_566_l0.001327_0.002605_0.000922_dr0.483312_m34.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu11_d1391_964_699_dr0.418089_m46.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu12_d1225_705_535_dr0.387340_m30.csv')
sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu10_d1129_913_597_l0.001868_0.000202_0.001368_dr0.412037_m47.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu13_d1086_918_355_dr0.469160_m43.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu11_d1018_779_564_l0.000826_0.002645_0.002176_dr0.328935_m41.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu14_d1431_936_628_dr0.399226_m47.csv')
sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu12_d1136_879_611_l0.000523_0.002560_0.002216_dr0.420966_m40.csv')
sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu15_d1061_983_333_dr0.482943_m36.csv')
sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu13_d1046_875_369_l0.002764_0.002404_0.001339_dr0.437976_m30.csv')
sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu16_d1301_725_643_dr0.373816_m34.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu14_d1454_955_370_l0.001240_0.001700_0.000135_dr0.367196_m44.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu17_d1254_815_474_dr0.332982_m39.csv')
sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu18_d1085_816_370_dr0.414340_m49.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu15_d1176_927_473_l0.000929_0.002994_0.002942_dr0.332054_m50.csv')
sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu19_d1083_933_313_dr0.464837_m40.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu16_d1390_976_481_l0.000517_0.001012_0.001913_dr0.390623_m32.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu20_d1334_905_400_dr0.402574_m32.csv')
sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu21_d1030_864_373_dr0.436626_m40.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu17_d1142_989_607_l0.000756_0.002179_0.002536_dr0.483245_m42.csv')
sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu22_d1016_960_303_dr0.487391_m36.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu18_d1234_809_504_l0.002821_0.001704_0.002203_dr0.365931_m32.csv')
sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu19_d1402_984_347_l0.000106_0.001493_0.002565_dr0.482983_m31.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu23_d1132_862_422_dr0.404973_m48.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu24_d1069_928_690_dr0.384504_m36.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu20_d1476_830_354_l0.002890_0.000136_0.002883_dr0.319011_m36.csv')
sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu25_d1414_843_314_dr0.305738_m30.csv')
sub49 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu21_d1298_845_321_l0.000440_0.001661_0.000876_dr0.371946_m45.csv')
sub50 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu26_d1115_978_331_dr0.374231_m45.csv')
sub51 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu22_d1465_707_599_l0.001096_0.002822_0.001441_dr0.300278_m40.csv')
sub52 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu27_d1394_939_449_dr0.425230_m35.csv')
sub53 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu23_d1319_917_574_l0.001725_0.001579_0.002241_dr0.378026_m40.csv')
sub54 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu28_d1101_945_654_dr0.377536_m33.csv')
sub55 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu24_d1021_775_590_l0.002293_0.001760_0.000215_dr0.452877_m40.csv')
sub56 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu29_d1282_705_676_dr0.302235_m42.csv')
sub57 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu25_d1065_951_519_l0.000189_0.002517_0.001186_dr0.339849_m38.csv')
sub58 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu30_d1403_755_607_dr0.429843_m44.csv')
sub59 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu31_d1010_942_638_dr0.389380_m31.csv')
sub60 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu26_d1339_962_302_l0.001055_0.001471_0.000807_dr0.327472_m44.csv')
sub61 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu32_d1295_789_343_dr0.300664_m39.csv')
sub62 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu27_d1493_969_351_l0.002376_0.002022_0.001774_dr0.321871_m39.csv')
sub63 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu33_d1335_707_537_dr0.369238_m49.csv')
sub64 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu28_d1012_833_578_l0.000842_0.002647_0.002283_dr0.495197_m43.csv')
sub65 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu34_d1159_956_549_dr0.311058_m37.csv')
sub66 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu29_d1462_782_682_l0.000905_0.001486_0.001032_dr0.412653_m34.csv')
sub67 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu35_d1016_987_630_dr0.436989_m39.csv')
sub68 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu30_d1039_869_583_l0.000901_0.002004_0.001912_dr0.434204_m44.csv')
sub69 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu36_d1368_952_676_dr0.385644_m43.csv')
sub70 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu31_d1204_714_690_l0.000824_0.000920_0.002488_dr0.370516_m43.csv')
sub71 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu37_d1311_805_574_dr0.381639_m46.csv')
sub72 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu32_d1017_757_629_l0.000345_0.000562_0.000583_dr0.458526_m31.csv')
sub73 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu38_d1021_995_464_dr0.474186_m31.csv')
sub74 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu33_d1394_938_388_l0.002957_0.001371_0.000960_dr0.311789_m40.csv')
sub75 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu39_d1330_923_447_dr0.348065_m39.csv')
sub76 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu34_d1447_939_319_l0.000264_0.000229_0.002585_dr0.455973_m34.csv')
sub77 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu40_d1099_741_590_dr0.324780_m32.csv')
sub78 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu35_d1485_843_577_l0.001068_0.000340_0.000733_dr0.483533_m31.csv')
sub79 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu41_d1271_704_514_dr0.473447_m42.csv')
sub80 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu36_d1025_992_314_l0.002585_0.001530_0.002390_dr0.309386_m31.csv')
sub81 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu42_d1008_708_483_dr0.383804_m45.csv')
sub82 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu43_d1008_870_466_dr0.466902_m38.csv')
sub83 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu37_d1169_778_655_l0.001601_0.000815_0.001472_dr0.470229_m48.csv')
sub84 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu44_d1127_794_339_dr0.492221_m41.csv')
sub85 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu38_d1032_925_378_l0.001152_0.000882_0.001061_dr0.332161_m41.csv')
sub86 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu45_d1290_764_479_dr0.490823_m34.csv')
sub87 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu39_d1487_845_385_l0.001481_0.002695_0.001619_dr0.466549_m37.csv')
sub88 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu46_d1226_815_677_dr0.374156_m38.csv')
sub89 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu40_d1120_938_388_l0.000199_0.002807_0.002751_dr0.406093_m40.csv')
sub90 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu47_d1477_838_652_dr0.448004_m39.csv')
sub91 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu41_d1262_773_505_l0.002222_0.001468_0.001254_dr0.312950_m35.csv')
sub92 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu48_d1184_984_558_dr0.434942_m38.csv')
sub93 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/relu49_d1127_986_338_dr0.340868_m30.csv')
sub94 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu42_d1210_871_421_l0.002575_0.001019_0.002751_dr0.347308_m39.csv')
sub95 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu43_d1278_948_582_l0.002272_0.002733_0.002195_dr0.337844_m38.csv')
sub96 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu44_d1428_765_524_l0.001962_0.001881_0.000425_dr0.485006_m32.csv')
sub97 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu45_d1046_864_453_l0.000969_0.002053_0.002077_dr0.366084_m46.csv')
sub98 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu46_d1392_744_649_l0.001497_0.001673_0.002746_dr0.477995_m43.csv')
sub99 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu47_d1225_974_404_l0.000342_0.000382_0.000808_dr0.339791_m50.csv')
sub100 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu48_d1342_957_317_l0.001021_0.000198_0.000847_dr0.390086_m49.csv')
sub101 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu49_d1214_863_637_l0.002350_0.000907_0.002721_dr0.401939_m37.csv')



final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub20+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub30+sub31+sub32+sub33+sub34+sub35+sub36+sub37+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48+sub49+sub50+sub51+sub52+sub53+sub54+sub55+sub56+sub57+sub58+sub59+sub60+sub61+sub62+sub63+sub64+sub65+sub66+sub67+sub68+sub69+sub70+sub71+sub72+sub73+sub74+sub75+sub76+sub77+sub78+sub79+sub80+sub81+sub82+sub83+sub84+sub85+sub86+sub87+sub88+sub89+sub90+sub91+sub92+sub93+sub94+sub95+sub96+sub97+sub98+sub99+sub100+sub101)/101.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/Fsub2.csv',index = False)



