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
        if nn.max_epochs % 10 == 0:
            epoch = train_history[-1]['epoch']
            new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


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
dense_list = [random.randint(900, 1200),random.randint(600, 900),random.randint(200, 550),random.randint(200, 450)]
leakness_list = [random.uniform(.0001, .003),random.uniform(.0001, .003),random.uniform(.0001, .003),random.uniform(.0001, .003)]
max_iter = random.randint(35,55)
#dropout_3 = random.uniform(.3, .5)
print dense_list
print leakness_list
print max_iter
    #print dropout_3

    net0 = NeuralNet(layers=layers0,
                     
                     input_shape=(None, num_features),
                     dropout_in_p=0.2,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     dense3_num_units=dense_list[3],
                     dropout3_p=0.5,
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
                    dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
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

                    max_epochs=95)
    net0.fit(X, y)
    submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada%i_d%i_%i_%i_l%f_%f_%f_dr0.2_m%i.csv' % (i,dense_list[0],dense_list[1],dense_list[2],leakness_list[0],leakness_list[1],leakness_list[2],max_iter2)
    make_submission(net0, X_test, ids, encoder, name=submission_name)




for i in xrange(0,50):
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
               #('dense4', DenseLayer),
               #('dropout4', DropoutLayer),
               ('output', DenseLayer)]
    dense_list = [random.randint(900, 1200),random.randint(600, 900),random.randint(200, 550),random.randint(200, 450)]
    leakness_list = [random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005)]
    max_iter = random.randint(100,170)
    dropout_in = round(random.uniform(.1, .3),2)
    print dense_list
    print leakness_list
    print max_iter
    print dropout_in
        
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     dense3_num_units=dense_list[3],
                     dropout3_p=0.5,
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
                    dense1_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[1]),
                    dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
                    dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
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
                    batch_iterator_train=BatchIterator(batch_size=128),
                    batch_iterator_test=BatchIterator(batch_size=128 ),

                    max_epochs=max_iter)
    net0.fit(X, y)
    val_loss = []
    for j in xrange(0,max_iter):
        val_loss.append(net0.train_history_[j]['valid_loss'])
    if min(val_loss) < 0.47:
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
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     dense3_num_units=dense_list[3],
                     dropout3_p=0.5,
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
                    dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
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

        submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada%i_%f_d%i_%i_%i_%i_l%f_%f_%f_%f_dr%f_m%i.csv' % (i,min_loss,dense_list[0],dense_list[1],dense_list[2],dense_list[3],leakness_list[0],leakness_list[1],leakness_list[2],leakness_list[3],dropout_in,max_iter2)
        make_submission(net0, X_test, ids, encoder, name=submission_name)

for i in xrange(0,50):
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
    dense_list = [random.randint(700, 1000),random.randint(500, 800),random.randint(300, 450),random.randint(200, 350),random.randint(200, 450)]
    leakness_list = [random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005),random.uniform(.0001, .0005)]
    max_iter = random.randint(150,180)
    dropout_in = round(random.uniform(.1, .3),2)
    print dense_list
    print leakness_list
    print max_iter
    print dropout_in
        
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     dense3_num_units=dense_list[3],
                     dropout3_p=0.5,
                     dense4_num_units=dense_list[4],
                     dropout4_p=0.5,
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
                    dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
                    dense4_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[4]),
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
                    batch_iterator_train=BatchIterator(batch_size=156),
                    batch_iterator_test=BatchIterator(batch_size=156 ),

                    max_epochs=max_iter)
    net0.fit(X, y)
    val_loss = []
    for j in xrange(0,max_iter):
        val_loss.append(net0.train_history_[j]['valid_loss'])
    if min(val_loss) < 0.475:
        max_iter2 = pd.Series(val_loss).argsort()[0] + 10

        net0 = NeuralNet(layers=layers0,                     
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
                     #dense2_nonlinearity=sigmoid,
                     dense3_num_units=dense_list[3],
                     dropout3_p=0.5,
                     dense4_num_units=dense_list[4],
                     dropout4_p=0.5,
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
                    dense3_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[3]),
                    dense4_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[4]),
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

        submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada%i_%f_d%i_%i_%i_%i_l%f_%f_%f_%f_dr%f_m%i.csv' % (i,min_loss,dense_list[0],dense_list[1],dense_list[2],dense_list[3],leakness_list[0],leakness_list[1],leakness_list[2],leakness_list[3],dropout_in,max_iter2)
        make_submission(net0, X_test, ids, encoder, name=submission_name)



for i in xrange(0,50):
    layers0 = [('input', InputLayer),
            ('dropout_in', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('dense2', DenseLayer),
               ('dropout2', DropoutLayer),
               #('dense3', DenseLayer),
               #('dropout3', DropoutLayer),
               ('output', DenseLayer)]
    dense_list = [random.randint(900, 1300),random.randint(600, 1000),random.randint(200, 750),random.randint(200, 450)]
    leakness_list = [random.uniform(.0001, .003),random.uniform(.0001, .003),random.uniform(.0001, .003),random.uniform(.0001, .003)]
    max_iter = random.randint(70,110)
    dropout_in = round(random.uniform(.1, .5),2)
    print dense_list
    print leakness_list
    print max_iter
    print dropout_in
        
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout_in_p=dropout_in,
                     dense0_num_units=dense_list[0],
                     dropout0_p=0.5,
                     #dense0_nonlinearity=sigmoid,
                     dense1_num_units=dense_list[1],
                     dropout1_p=0.5,
                     #dense1_nonlinearity=sigmoid,
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
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
                     dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
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
                    eval_size=0.2,
                    verbose=1,
                    batch_iterator_train=BatchIterator(batch_size=128),
                    batch_iterator_test=BatchIterator(batch_size=128 ),
                    max_epochs=max_iter)
    net0.fit(X, y)
    val_loss = []

    for j in xrange(0,max_iter):
        val_loss.append(net0.train_history_[j]['valid_loss'])
    if min(val_loss)<0.46:
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
                     dense2_num_units=dense_list[2],
                     dropout2_p=0.5,
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
                    dense2_nonlinearity= nonlin.LeakyRectify(leakiness=leakness_list[2]),
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
        val_loss = []

        for j in xrange(0,max_iter2):
            val_loss.append(net0.train_history_[j]['valid_loss'])
        min_loss = min(val_loss)
        submission_name = '/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada%i_%f_d%i_%i_%i_l%f_%f_%f_dr%f_m%i.csv' % (i,min_loss,dense_list[0],dense_list[1],dense_list[2],leakness_list[0],leakness_list[1],leakness_list[2],dropout_in,max_iter2)
        make_submission(net0, X_test, ids, encoder, name=submission_name)







lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95
0.44792



X_rand = np.random.normal(size=X.shape)*0.1
X + X_rand

random.gauss(0,1)











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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')
sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')
#sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission17_sd_count_md.csv')
#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission18_sd_count_md.csv')
#sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission19_sd_count_md.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission20_sd_count_md.csv')
#submission1と2を使ってみる. 3は化学臭すぎ?
#final_sub3 0.41623


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub30)/27.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop1.csv',index = False)

final_sub4 neural_net_sub1_all_countなし
0.41618

final_sub4_neuraldrop1
0.41546
'''


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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')

sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0_d1195_865_237_418_l0.000716_0.000424_0.002197_0.001230_dr0.253276_m124.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1229_918_466_l0.000285_0.002802_0.002276_dr0.190000_m69.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada4_0_d947_683_533_l0.002666_0.000902_0.000748_dr0.220000_m99.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub30+sub31+sub32+sub33+sub34+sub35+sub36+sub37+sub38)/35.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop2.csv',index = False)


final_sub4_neuraldrop2
0.41595
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')

sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0_d1195_865_237_418_l0.000716_0.000424_0.002197_0.001230_dr0.253276_m124.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1229_918_466_l0.000285_0.002802_0.002276_dr0.190000_m69.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
#sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada4_0_d947_683_533_l0.002666_0.000902_0.000748_dr0.220000_m99.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub30+sub31+sub33+sub34+sub35+sub36+sub38)/33.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop3.csv',index = False)


final_sub4_neuraldrop3
0.41545
'''
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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')

sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0_d1195_865_237_418_l0.000716_0.000424_0.002197_0.001230_dr0.253276_m124.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
#sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1229_918_466_l0.000285_0.002802_0.002276_dr0.190000_m69.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
#sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada4_0_d947_683_533_l0.002666_0.000902_0.000748_dr0.220000_m99.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub31+sub33+sub34+sub36+sub38)/31.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop4.csv',index = False)


final_sub4_neuraldrop4
0.41521
'''
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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')

sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0_d1195_865_237_418_l0.000716_0.000424_0.002197_0.001230_dr0.253276_m124.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
#sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1229_918_466_l0.000285_0.002802_0.002276_dr0.190000_m69.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
#sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada4_0_d947_683_533_l0.002666_0.000902_0.000748_dr0.220000_m99.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub31+sub33+sub34+sub36+sub38+sub39)/32.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop5.csv',index = False)


final_sub4_neuraldrop4
0.41521
'''


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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')

sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0_d1195_865_237_418_l0.000716_0.000424_0.002197_0.001230_dr0.253276_m124.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
#sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1229_918_466_l0.000285_0.002802_0.002276_dr0.190000_m69.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
#sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada4_0_d947_683_533_l0.002666_0.000902_0.000748_dr0.220000_m99.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')
sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada13_0_d957_692_502_l0.002106_0.002093_0.000147_dr0.150000_m110.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada7_0_d1256_979_552_l0.002036_0.000699_0.001511_dr0.160000_m96.csv')
sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada6_0_d902_740_529_l0.002875_0.002485_0.000554_dr0.100000_m78.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada5_0.407126_d1144_828_205_247_l0.000333_0.000273_0.000129_0.001682_dr0.149717_m119.csv')
sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada4_0.434769_d1001_683_506_213_l0.000498_0.000341_0.000254_0.002425_dr0.263006_m169.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada3_0.419438_d1055_671_502_416_l0.000384_0.000298_0.000124_0.002805_dr0.148855_m114.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.404749_d908_888_320_401_l0.000445_0.000383_0.000369_0.001484_dr0.149769_m118.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada0_0.410101_d1070_642_391_283_l0.000374_0.000488_0.000261_0.001681_dr0.137166_m144.csv')



final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub31+sub33+sub34+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47)/40.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop6.csv',index = False)


final_sub4_neuraldrop6
0.41593
'''
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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

#sub30 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d971_936_741_l0.001203_0.002983_0.002241_dr0.147353_m86.csv')

#sub31 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
#sub32 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0_d1195_865_237_418_l0.000716_0.000424_0.002197_0.001230_dr0.253276_m124.csv')
sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
#sub35 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1229_918_466_l0.000285_0.002802_0.002276_dr0.190000_m69.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
#sub37 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada4_0_d947_683_533_l0.002666_0.000902_0.000748_dr0.220000_m99.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

#sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada13_0_d957_692_502_l0.002106_0.002093_0.000147_dr0.150000_m110.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada7_0_d1256_979_552_l0.002036_0.000699_0.001511_dr0.160000_m96.csv')
sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada6_0_d902_740_529_l0.002875_0.002485_0.000554_dr0.100000_m78.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada5_0.407126_d1144_828_205_247_l0.000333_0.000273_0.000129_0.001682_dr0.149717_m119.csv')
#sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada4_0.434769_d1001_683_506_213_l0.000498_0.000341_0.000254_0.002425_dr0.263006_m169.csv')
#sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada3_0.419438_d1055_671_502_416_l0.000384_0.000298_0.000124_0.002805_dr0.148855_m114.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.404749_d908_888_320_401_l0.000445_0.000383_0.000369_0.001484_dr0.149769_m118.csv')
#sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada0_0.410101_d1070_642_391_283_l0.000374_0.000488_0.000261_0.001681_dr0.137166_m144.csv')



final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub31+sub33+sub34+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub46)/37.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop7.csv',index = False)


final_sub4_neuraldrop6
0.41593
'''

sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')
sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada1_0_d1020_617_484_l0.002876_0.001514_0.000229_dr0.110000_m88.csv')
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada7_0_d1256_979_552_l0.002036_0.000699_0.001511_dr0.160000_m96.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada6_0_d902_740_529_l0.002875_0.002485_0.000554_dr0.100000_m78.csv')
sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada5_0.407126_d1144_828_205_247_l0.000333_0.000273_0.000129_0.001682_dr0.149717_m119.csv')
sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.404749_d908_888_320_401_l0.000445_0.000383_0.000369_0.001484_dr0.149769_m118.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11)/11.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/Submission414.csv',index = False)

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub34 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada0_0_d1252_674_360_l0.002885_0.000736_0.002167_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub33+sub34+sub36+sub38+sub39)/31.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop7.csv',index = False)


final_sub4_neuraldrop7
0.41502
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub33+sub36+sub38+sub39)/30.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop8.csv',index = False)


final_sub4_neuraldrop8
0.41497
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

#sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
#sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub36+sub38)/28.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop9.csv',index = False)


final_sub4_neuraldrop9
0.41531
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub33+sub36+sub38+sub39+sub40+sub41)/32.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop10.csv',index = False)

final_sub4_neuraldrop10
0.41483
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43)/34.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop11.csv',index = False)

final_sub4_neuraldrop11
0.41442
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub25+sub26+sub27+sub28+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47)/38.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop12.csv',index = False)

final_sub4_neuraldrop12
0.41419
'''

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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47)/34.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop13.csv',index = False)

final_sub4_neuraldrop13
0.41403
'''


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

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')


final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47)/28.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop14.csv',index = False)

final_sub4_neuraldrop14
0.41439
'''
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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48)/35.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop15.csv',index = False)

final_sub4_neuraldrop15
0.41401
'''
'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
#sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
#sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
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
#sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
#sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48)/31.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop16.csv',index = False)

final_sub4_neuraldrop16
0.41469
'''
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
#sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
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

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48)/34.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop17.csv',index = False)

final_sub4_neuraldrop17
0.41402
'''
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
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

sub49 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada0_0.429716_d952_620_250_242_l0.000257_0.000368_0.000257_0.000313_dr0.250000_m159.csv')
sub50 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada0_0.432426_d881_624_406_307_l0.000245_0.000266_0.000412_0.000237_dr0.150000_m171.csv')

final_sub = (sub3+sub4+sub5+sub6+sub7+sub8+sub9+sub10+sub11+sub12+sub13+sub14+sub15+sub16+sub17+sub18+sub19+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48+sub49+sub50)/37.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop18.csv',index = False)

final_sub4_neuraldrop18
0.41402
'''

'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
#sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
#sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
#sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
#sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
#sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
#sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

final_sub = (sub3+sub4+sub7+sub8+sub9+sub12+sub13+sub14+sub15+sub16+sub17+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48)/29.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop20.csv',index = False)

final_sub4_neuraldrop20
0.41329
'''

'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
#sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
#sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
#sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
#sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
#sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
#sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

sub49 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgblrelu3ada0_0.300472_d1325_718_363_l0.000126_0.000422_0.000296_dr0.250000_m95.csv')
sub50 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.448576_d732_569_363_244_l0.000165_0.000434_0.000283_0.000259_dr0.110000_m167.csv')
sub51 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgblrelu3ada1_0.318927_d1061_869_374_l0.000357_0.000376_0.000372_dr0.210000_m71.csv')
sub52 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.403742_d921_632_387_300_l0.000114_0.000113_0.000250_0.000112_dr0.120000_m159.csv')
sub53 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgblrelu3ada2_0.316078_d1368_805_482_l0.000167_0.000322_0.000169_dr0.250000_m89.csv')
sub54 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgblrelu3ada3_0.330521_d1171_990_539_l0.000277_0.000347_0.000459_dr0.200000_m76.csv')
sub55 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada4_0.441028_d751_611_344_313_l0.000175_0.000249_0.000212_0.000163_dr0.110000_m137.csv')

final_sub = (sub3+sub4+sub7+sub8+sub9+sub12+sub13+sub14+sub15+sub16+sub17+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48+sub49+sub50+sub51+sub52+sub53+sub54+sub55)/36.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop21.csv',index = False)

final_sub4_neuraldrop21
0.41369
'''

'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
#sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
#sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
#sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
#sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
#sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
#sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
#sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
#sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
#sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
#sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

final_sub = (sub3+sub4+sub7+sub8+sub9+sub12+sub13+sub14+sub15+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48)/25.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop23.csv',index = False)

final_sub4_neuraldrop23
0.41400
'''
'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
#sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
#sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
#sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
#sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
#sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
#sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
#sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
#sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
#sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
#sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

final_sub = (sub3+sub4+sub12+sub14+sub15+sub16+sub17+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48)/25.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop24.csv',index = False)

final_sub4_neuraldrop24
0.41330
'''
'''
sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')
sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd_count_md.csv')
#sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd_count_md.csv')
#sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd_count_md.csv')
sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')
sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all.csv')
sub9 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all.csv')
#sub10 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all.csv')
#sub11 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all.csv')
sub12 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')
sub13 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')
sub14 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')
sub15 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission5_sd.csv')
sub16 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd.csv')
sub17 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission7_sd.csv')
#sub18 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission8_sd.csv')
#sub19 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission9_sd.csv')
sub21 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')
sub22 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count.csv')
sub23 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count.csv')
sub24 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub5_all_count.csv')

#sub25 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all_count_lrelu.csv')
#sub26 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count_lrelu.csv')
#sub27 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub3_all_count_lrelu.csv')
#sub28 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub4_all_count_lrelu.csv')

sub29 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub33 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada2_0_d983_945_486_l0.000378_0.001621_0.001404_dr0.180000_m97.csv')
sub36 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada3_0_d1082_947_453_l0.000431_0.000841_0.001840_dr0.220000_m102.csv')
sub38 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada12_0_d1000_899_252_435_l0.001384_0.001475_0.000224_0.002142_dr0.181354_m98.csv')
sub39 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu3ada5_0_d1202_881_399_l0.000492_0.002107_0.000158_dr0.190000_m104.csv')

sub40 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada2_0.400060_d959_828_509_376_l0.000444_0.000421_0.000378_0.000183_dr0.210000_m165.csv')
sub41 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lrelu4ada1_0.386414_d988_863_528_315_l0.000418_0.000252_0.000357_0.000421_dr0.180000_m143.csv')

sub42 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')
sub43 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count2.csv')

sub44 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count3.csv')
sub45 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count4.csv')
sub46 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count5.csv')
sub47 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count6.csv')

sub48 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count7.csv')

sub49 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count8.csv')
sub50 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count9.csv')
sub51 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count10.csv')
sub52 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count11.csv')
sub53 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count13.csv')
sub54 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count15.csv')
sub55 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count16.csv')
sub56 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count17.csv')
sub58 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count19.csv')
sub59 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count20.csv')
sub60 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count21.csv')
sub61 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count22.csv')
sub62 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count23.csv')
sub63 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count24.csv')
sub64 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count25.csv')
sub65 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count26.csv')

final_sub = (sub3+sub4+sub7+sub8+sub9+sub12+sub13+sub14+sub15+sub16+sub17+sub21+sub22+sub23+sub24+sub29+sub33+sub36+sub38+sub39+sub40+sub41+sub42+sub43+sub44+sub45+sub46+sub47+sub48+sub49+sub50+sub51+sub52+sub53+sub54+sub55+sub56+sub58+sub59+sub60+sub61+sub62+sub63+sub64+sub65)/45.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop25.csv',index = False)

final_sub4_neuraldrop25
0.41890
'''
'''
sub1 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission6_sd_count_md.csv')

sub2 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub1_all.csv')

sub3 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb1.csv')

sub4 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission10_sd_count.csv')

sub5 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/submission_sd.csv')

sub6 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/neural_net_sub2_all_count.csv')

sub7 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/lreluada0_d1087_774_480_306_l0.001308_0.001160_0.001950_0.000727_dr0.2_m95.csv')

sub8 = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/xgb_count1.csv')

final_sub = (sub1+sub2+sub3+sub4+sub5+sub6+sub7+sub8)/8.0
final_sub['id'] = final_sub['id'].astype(int)
final_sub.to_csv('/Users/IkkiTanaka/Documents/kaggle/Otto/final_sub4_neuraldrop26.csv',index = False)

final_sub4_neuraldrop26
0.41329
'''

