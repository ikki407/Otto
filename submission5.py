import graphlab as gl
import math
import random
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
train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_count.csv')
test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_count.csv')
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
train = shuffle(train)




params = {'target': 'target',
          'max_iterations': 350,
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}


#パラメータのdefault調べる  stepsize = 
train = shuffle(train)

# Check performance on internal validation set
#一旦、valでだいたいの値を評価してから、全trainデータで評価する。
tr, va = train.random_split(.8)
m = gl.boosted_trees_classifier.create(tr, **params)

print evaluate_logloss(m, tr, va)

# Make final submission by using full training set
m = gl.boosted_trees_classifier.create(train, **params)

make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission1_sd_count.csv')

'''

#submission_sd_count
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd_count.csv')


#submission5_sd_count
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .6,
          'min_loss_reduction': 1,
          'column_subsample': .75,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count.csv')


#submission6_sd_count
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count.csv')



#submission7_sd_count
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 8,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count.csv')



#submission8_sd_count
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count.csv')



#submission9_sd_count 
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count.csv')

#submission10_sd_count

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')


sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd_count.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')

#submission1と2を使ってみる. 3は化学臭すぎ?


final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12)/12.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sd_count_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_xgb1.csv',index = False)


sd_count_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_xgb1

'''

'''
#submission10_sd_count

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')

#submission_sd_count_md
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 16,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd_count_md.csv')


#submission5_sd_count_md

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 16,
          'min_child_weight': 4,
          'row_subsample': .6,
          'min_loss_reduction': 1,
          'column_subsample': .75,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')


#submission6_sd_count_md
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 14,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')



#submission7_sd_count_md
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 14,
          'min_child_weight': 3,
          'row_subsample': .7,
          'min_loss_reduction': 0.8,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')



#submission8_sd_count_md
params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 16,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'step_size': 0.2,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')



#submission9_sd_count_md
params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 16,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .9,
          'step_size': 0.1,
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')

#submission10_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 20,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count_md.csv')

#submission11_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 20,
          'min_child_weight': 5,
          'row_subsample': .7,
          'min_loss_reduction': 1.5,
          'column_subsample': .6,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission11_sd_count_md.csv')

#submission12_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 25,
          'min_child_weight': 4,
          'row_subsample': .5,
          'min_loss_reduction': 1.,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission12_sd_count_md.csv')

#submission13_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 25,
          'min_child_weight': 4,
          'row_subsample': .8,
          'min_loss_reduction': 1.,
          'column_subsample': .7,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission13_sd_count_md.csv')

#submission14_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 27,
          'min_child_weight': 4,
          'row_subsample': .6,
          'min_loss_reduction': 1.,
          'column_subsample': .6,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission14_sd_count_md.csv')

#submission15_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 26,
          'min_child_weight': 4,
          'row_subsample': .8,
          'min_loss_reduction': 1.,
          'column_subsample': .8,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission15_sd_count_md.csv')

#submission16_sd_count_md

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 26,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1.,
          'column_subsample': .6,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission16_sd_count_md.csv')



#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub20 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#submission1と2を使ってみる. 3は化学臭すぎ?


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub20+sub21+sub22+sub23+sub24)/22.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/sd_sub09_count_md_sub6_16__sub10_neural_all_1__5_xgb1_net15.csv',index = False)

'''
sd_count_md_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5_xgb1
0.42151

sd_count_md_sub_sub5_sub6_sub7_sub8_sub9_neural_all_1__5
0.42204

sd_sub_sd_count_md_sub5_sub6_sub7_sub8_sub9_sub10_neural_all_1__5_xgb1
0.42078

sd_sub59_count_md_sub0_sub5_sub6_sub7_sub8_sub9__sub10_neural_all_1__5_xgb1
0.42071

sd_sub059_count_md_sub5_sub6_sub7_sub8_sub9__sub10_neural_all_1__5_xgb1
0.42048

sd_sub09_count_md_sub6_sub7_sub8_sub9__sub10_neural_all_1__5_xgb1
0.42033

sd_sub09_count_md_sub7_sub8_sub9__sub10_neural_all_1__5_xgb1
0.42037

sd_sub09_count_md_sub6_16__sub10_neural_all_1__5_xgb1
0.42104

sd_sub09_count_md_sub6_9_sub10_neural_all_1__5_xgb1_net15
0.41632
'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub20 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

2-layerのneural_net_sub10_all_countは使用しないほうが良いかも
'''
train = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/train_sd_count_only.csv')
test = gl.SFrame.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/test_sd_count_only.csv')
del train['id']
train = shuffle(train)
tr, va = train.random_split(.8)


params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 26,
          'min_child_weight': 4,
          'row_subsample': .75,
          'min_loss_reduction': 1.,
          'column_subsample': .6,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
#m = gl.boosted_trees_classifier.create(tr, **params)
#print evaluate_logloss(m, tr, va)

m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 28,
          'min_child_weight': 3,
          'row_subsample': .8,
          'min_loss_reduction': 1.,
          'column_subsample': .7,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
#m = gl.boosted_trees_classifier.create(tr, **params)
#print evaluate_logloss(m, tr, va)

m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 28,
          'min_child_weight': 3,
          'row_subsample': .5,
          'min_loss_reduction': 1.,
          'column_subsample': .5,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
#m = gl.boosted_trees_classifier.create(tr, **params)
#print evaluate_logloss(m, tr, va)

m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 23,
          'min_child_weight': 3,
          'row_subsample': .6,
          'min_loss_reduction': 1.,
          'column_subsample': .6,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
#m = gl.boosted_trees_classifier.create(tr, **params)
#print evaluate_logloss(m, tr, va)

m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')

params = {'target': 'target',
          'max_iterations': 300,
          'max_depth': 22,
          'min_child_weight': 3,
          'row_subsample': .6,
          'min_loss_reduction': 1.,
          'column_subsample': .4,
          'step_size': 0.2,
          #'class_weights' : 'auto',
          'validation_set': None}
#m = gl.boosted_trees_classifier.create(tr, **params)
#print evaluate_logloss(m, tr, va)

m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, '/Users/IkkiTanaka/Documents/kaggle/Otto/submission21_sd_count_md.csv')

'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub20 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')
#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission21_sd_count_md.csv')

sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub6_all_count.csv')
sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub7_all_count.csv')
sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub8_all_count.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub9_all_count.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub10_all_count.csv')
sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub11_all_count.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub12_all_count.csv')
sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub13_all_count.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub14_all_count.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub15_all_count.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')


#submission1と2を使ってみる. 3は化学臭すぎ?
#final_sub1はpcaのNN
#final_sub2
0.41869


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub20+sub21+sub22+sub23+sub24+sub30+sub31+sub32+sub33+sub34+sub35+sub36+sub37+sub38+sub39+sub40+sub41+sub42+sub43)/36.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub2.csv',index = False)
'''



'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub20 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')


#submission1と2を使ってみる. 3は化学臭すぎ?
#final_sub3
0.41623

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub20+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28)/26.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub3.csv',index = False)
'''
20~28を1個ずつみろ！！！

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')
#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission21_sd_count_md.csv')
'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')
#submission1と2を使ってみる. 3は化学臭すぎ?
#final_sub3 0.41623


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28)/25.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub8.csv',index = False)
final_sub4 neural_net_sub1_all_countなし
0.41618
final_sub5 neural_net_sub1_all_count,neural_net_sub5_all_countなし
0.41626
final_sub6 neural_net_sub1_all_count_lreluなし
0.41620




'''

'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_pca.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_pca.csv')
sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_pca.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_pca.csv')
sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count_pca.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_pca_lrelu.csv')
sub49 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_pca_lrelu.csv')
sub50 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_pca_lrelu.csv')
sub51 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_pca_lrelu.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub33+sub34+sub35+sub36+sub37+sub48+sub49+sub50+sub51)/34.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub7.csv',index = False)

final_sub7
0.41906

0.41724

'''

'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')
#submission1と2を使ってみる. 3は化学臭すぎ?
#final_sub3 0.41623

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN2_4.csv')
sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb2_NN2_4.csv')
sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb3_NN2_4.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub30+sub31)/28.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub9.csv',index = False)

final_sub9
0.41643


'''



'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
#sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
#sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
#sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
#sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
#sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
#sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
#sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')
#submission1と2を使ってみる. 3は化学臭すぎ?
#final_sub3 0.41623

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_NN2_4.csv')
sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb2_NN2_4.csv')
sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb3_NN2_4.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28)/18.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub10.csv',index = False)

final_sub10
0.41938

'''
'''
#sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd_count_md.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28)/34.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub11.csv',index = False)




