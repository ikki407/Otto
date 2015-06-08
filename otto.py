from csv import DictReader
from os.path import join, expanduser
from math import log, copysign
import gzip
import itertools

_datapath = '/Users/IkkiTanaka/Documents/kaggle/Otto/'

def _data(dataset):
    """
    Read the datafile and return the Id, features and outcome.

    `features` is a dict of feature:value combinations.
    """

    for row in DictReader(open(dataset)):
        Id = row['id']
        # Ignore features that are zero
        features = {i:int(j) for i,j in row.iteritems() if i[:5] == 'feat_' and int(j) > 0}
        outcome = row['target'] if 'target' in row else None
        yield Id,features,outcome

def logloss(y_true, y_pred, classes):
    """ Log Loss from Kaggle. """
    idx = [list(classes).index(y) for y in y_true]
    logloss = sum([-log(max(min(y[i],1. - 1.e-15),1.e-15)) for y,i in zip(y_pred,idx)])
    return logloss / len(y_true)

def create_submit_file(clf, filename):
    with gzip.open(filename, 'wb') as fout:
        fout.write('id,')
        fout.write(','.join(clf.classes_))
        fout.write('\n')
        for Id,X,y in test_set:
            probs = {cls:pr for cls,pr in zip(clf.classes_,clf.predict_proba(X)[0])}
            fout.write('%s,' % Id)
            fout.write(','.join(['%0.4f' % probs[cls] for cls in clf.classes_]))
            fout.write('\n')

def class_means(Xin, yin):
    stats = {}
    for i,(X,y) in enumerate(itertools.izip(Xin,yin)):
        for feature in X.keys():
            stats[y,feature].append(X[feature] if feature in X else 0)

    means = {}
    for (y,feature),dat in stats.iteritems():
        means[y,feature] = 1. * sum(dat) / len(dat) if len(dat) > 0 else 0.
    return means

train_set = _data(join(_datapath, 'train.csv'))
test_set = _data(join(_datapath, 'test.csv'))
