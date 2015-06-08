from sklearn import ensemble, linear_model, cross_validation, svm, naive_bayes, neighbors, tree
import pandas as pd
import constants

def get_training_vectors():
    training_filename = constants.training_data_csv
    file_contents = pd.read_csv(training_filename, index_col=constants.ID_COLUMN)
    feature_columns, target_column = file_contents.columns[:-1], file_contents.columns[-1]
    return file_contents[feature_columns], file_contents[target_column]

def get_testing_vectors():
    testing_filename = constants.testing_data_csv
    file_contents = pd.read_csv(testing_filename, index_col=constants.ID_COLUMN)
    feature_columns= file_contents.columns
    return file_contents[feature_columns]

class CrossValidationRun(object):
    def run_cross_validation(self, model):
        training_input, training_output = get_training_vectors()
        cv_train_in, cv_test_in, cv_train_out, cv_test_out = cross_validation.train_test_split(training_input, training_output, test_size=0.7, random_state=0)
        model.fit(cv_train_in, cv_train_out)
        predictions = model.predict(cv_test_in)
        correct = sum([1 for predicted, gold in zip(predictions,cv_test_out) if predicted == gold])
        percent_correct = (1.0*correct/len(predictions)) * 100
        #print('Percent correct is {0}%'.format(percent_correct))
        print percent_correct


    def _split_list(self, L, k):
        buckets = [list() for i in range(k)]
        for element in L:
            key = hash(element) % k
            buckets[key].append(element)
        return buckets

class CompetitionRun(object):
    def train(self, model):
        training_input, training_output = get_training_vectors()
        self._fit_model(training_input, training_output, model)

    def _fit_model(self, training_input, training_output, model):
        model.fit(training_input, training_output)

    def test(self, model):
        testing_input = get_testing_vectors()
        predicted_output = self._predict_model(testing_input, model)
        self._write_output_to_file(testing_input.index, predicted_output)

    def _predict_model(self, testing_input, model):
        return model.predict(testing_input)

    def _write_output_to_file(self, index, predicted_output):
        f_out = open(constants.output_filename, 'w')
        f_out.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
        for id, predicted_class in zip(index, predicted_output):
            row = [str(id)] + self._vectorize_output_class(predicted_class)
            f_out.write(','.join(row) + '\n')
        f_out.close()

    def _vectorize_output_class(self, predicted_class):
        classes = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        return ['0.0' if c != predicted_class else '1.0' for c in classes]

if __name__ == '__main__':
    model = ensemble.RandomForestClassifier(n_estimators=1000)
    # model = ensemble.GradientBoostingClassifier() #??? Submission
    # model = svm.SVC()
    # model = linear_model.LogisticRegression()
    # model = linear_model.RidgeClassifier()
    #model = linear_model.SGDClassifier(penalty='l1')
    # model = neighbors.KNeighborsClassifier(weights='distance')

    cv = CrossValidationRun()
    cv.run_cross_validation(model)
    run = CompetitionRun()
    run.train(model)
    run.test(model)