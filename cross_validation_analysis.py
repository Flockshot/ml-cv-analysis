import numpy as np
from DataLoader import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score, \
    cross_validate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def calc_lower_bound(mean, std, n):
    return mean - ((1.96 * std) / np.sqrt(n))


def calc_upper_bound(mean, std, n):
    return mean + ((1.96 * std) / np.sqrt(n))


data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

methods = {"knn": KNeighborsClassifier, "svm": SVC, "tree": DecisionTreeClassifier, "forest": RandomForestClassifier}

forest_params = [{'n_estimators': n_estimators, 'criterion': criterion} for n_estimators in [100, 150] for criterion in
              ['gini', 'entropy']]

print(forest_params)
params = {
    'knn': {
        'knn__n_neighbors': [10, 15],
        'knn__metric': ['manhattan', 'cosine']
    },
    'svm': {
        'svm__kernel': ['rbf', 'sigmoid'],
        'svm__C': [10, 15]
    },
    'tree': {
        'tree__ccp_alpha': [0.01, 0.02],
        'tree__criterion': ['gini', 'entropy']
    },
    'forest': forest_params
}

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000))

print("Comparing Models on Accuracy: \n")

for method in methods.keys():
    for score in ["accuracy", "f1_micro"]:
        metrics = []
        minmax = MinMaxScaler(feature_range=(-1, 1))

        if method == "forest":


            for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
                current_training_part = dataset[train_indices]
                current_training_part_label = labels[train_indices]

                current_test_part = dataset[test_indices]
                current_test_part_label = labels[test_indices]


                forest_performance = dict()

                for inner_train_indices, inner_test_indices in inner_cross_validation.split(current_training_part, current_training_part_label):

                    inner_training_dataset = minmax.fit_transform(current_training_part[inner_train_indices])
                    inner_training_label = current_training_part_label[inner_train_indices]

                    inner_test_dataset = minmax.transform(current_training_part[inner_test_indices])
                    inner_test_label = current_training_part_label[inner_test_indices]


                    for param in params[method]:
                        param_str = str(param)


                        if str(param) not in forest_performance:
                            forest_performance[str(param)] = []


                        rand_scores = []
                        for i in range(10):
                            model = methods[method](**param)
                            model.fit(inner_training_dataset, inner_training_label)

                            predicted = model.predict(inner_test_dataset)

                            score_val = 0
                            if score == 'accuracy':
                                score_val = accuracy_score(inner_test_label, predicted)
                            else:
                                score_val = f1_score(inner_test_label, predicted, average='micro')

                            rand_scores.append(score_val)

                        mean_score = np.mean(rand_scores)
                        forest_performance[param_str].append(mean_score)

                best_parameter = None
                best_score = -float('inf')
                best_std = 0
                best_lower = 0
                best_upper = 0

                for param_config in forest_performance:
                    v = np.mean(forest_performance[param_config])
                    if v > best_score:
                        best_score = v
                        best_std = np.std(forest_performance[param_config])
                        best_lower = calc_lower_bound(best_score, best_std, len(forest_performance[param_config]))
                        best_upper = calc_upper_bound(best_score, best_std, len(forest_performance[param_config]))

                        best_parameter = param_config

                print(f"Params: {best_parameter}")
                print("Mean: %.2f" % best_score)
                print("Std: %.2f" % best_std)
                print("Lower Bound: %.2f" % best_lower)
                print("Upper Bound: %.2f" % best_upper)
                print("------------------------------------------------------------")

                current_training_part_minmax = minmax.fit_transform(current_training_part)
                current_test_part_minmax = minmax.transform(current_test_part)

                rand_scores = []
                for i in range(10):
                    model = methods[method](**eval(best_parameter))
                    model.fit(current_training_part_minmax, current_training_part_label)

                    predicted = model.predict(current_test_part_minmax)

                    score_val = 0
                    if score == 'accuracy':
                        score_val = accuracy_score(current_test_part_label, predicted)
                    else:
                        score_val = f1_score(current_test_part_label, predicted, average='micro')

                    rand_scores.append(score_val)

                mean = np.mean(rand_scores)
                metrics.append(mean)

        else:
            
            pipeline = Pipeline([('minmax', minmax), (method, methods[method]())])
            print(method)
            grid_search = GridSearchCV(pipeline, params[method], scoring=score, cv=inner_cross_validation, refit=True)

            metrics = cross_validate(grid_search, dataset, labels, scoring=score, cv=outer_cross_validation, return_estimator=True)

            for estimator in metrics['estimator']:
                best_i = estimator.best_index_
                mean = estimator.best_score_
                std = estimator.cv_results_['std_test_score'][best_i]
                lower_bound = calc_lower_bound(mean, std, len(estimator.cv_results_['mean_test_score']))
                upper_bound = calc_upper_bound(mean, std, len(estimator.cv_results_['mean_test_score']))
                best_params = estimator.best_params_

                print(f"Params: {best_params}")
                print("Mean: %.2f" % mean)
                print("Std: %.2f" % std)
                print("Lower Bound: %.2f" % lower_bound)
                print("Upper Bound: %.2f" % upper_bound)
                print("------------------------------------------------------------")

            metrics = metrics['test_score']

        mean = np.mean(metrics)
        std = np.std(metrics)
        lower_bound = calc_lower_bound(mean, std, len(metrics))
        upper_bound = calc_upper_bound(mean, std, len(metrics))

        print(f"Method: {method}")
        print("Mean: %.2f" % mean)
        print("Std: %.2f" % std)
        print("Lower Bound: %.2f" % lower_bound)
        print("Upper Bound: %.2f" % upper_bound)
        print("------------------------------------------------------------")

# evaluate_models(keys, scoring='accuracy')
