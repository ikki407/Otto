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

def load_train_data(df):
    #df = pd.read_csv(path)
    idx = range(1,len(df.columns))
    idx.remove(93)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:,idx].astype(np.float32), X[:,93]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    return X, y, encoder

def load_test_data(df):
    #df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    #X = scaler.transform(X)
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
del train['id']




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


X, y, encoder = load_train_data(train)
X_test, ids = load_test_data(test)
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
                 dense0_num_units=350,
                 dropout0_p=0.2,
                 dense1_num_units=750,
                 dropout1_p=0.3,
                 dense2_num_units=1050,
                 dropout2_p=0.4,
                 dense3_num_units=1350,
                 dropout3_p=0.5,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 #update_learning_rate=0.005,
                 #update_momentum=0.9,
                 update_learning_rate=theano.shared(float32(0.005)),
                 update_momentum=theano.shared(float32(0.9)),
    
                 on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.005, stop=0.00005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 ],

                 eval_size=0.1,
                 verbose=1,
                 max_epochs=65)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_submission9.csv')


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


'''
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
                 max_epochs=70)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder, name='/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')





sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1_3_4
0.42292

sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9)/9
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sd_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1_3_4.csv',index = False)



