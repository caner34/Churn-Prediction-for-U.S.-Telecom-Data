import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# DATA SOURCE:
# https://www.kaggle.com/becksddf/churn-in-telecoms-dataset


project_main_dir = '/home/a/Desktop/AnacondaProjects/Churn_Telecom/'


import sys
sys.path.append(project_main_dir)
import bidirectional_feature_selection as bfs
import upsampling


data = pd.read_csv(project_main_dir+'bigml_59c28831336c6604c800002a.csv')

print(data.columns.tolist())

data.churn = data.churn.apply(lambda x: 0 if x == False else 1)
data['international plan'] = data['international plan'].apply(lambda x: 0 if x == 'No' else 1)
data['voice mail plan'] = data['international plan'].apply(lambda x: 0 if x == 'No' else 1)
data = data.drop(columns=['phone number'])

print(data.churn.value_counts())


# Descriptive Analysis

data.describe()


# Create dummy variables for categorical feature
data['area code'] = data['area code'].astype(str)
cotegorical_columns = ['state', 'area code' ]
df_dummies = pd.get_dummies(data[cotegorical_columns])
data = pd.merge(data, df_dummies, how="inner", left_index=True, right_index=True).drop(columns=cotegorical_columns)


# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
column_list_for_scaling = data.columns.tolist()[:17]
data[column_list_for_scaling] = min_max_scaler.fit_transform(data[column_list_for_scaling])



X = data.drop(['churn'], axis = 1)
y = data.churn


# CLASSIFIER SELECTION PIPELINE
# https://www.kaggle.com/sandipdatta/customer-churn-analysis
# from "Stratified Cross Validation - Since the Response values are not balanced" on

# ensemble.GradientBoostingClassifier
# svm.SVC
# ensemble.RandomForestClassifier
# neighbors.KNeighborsClassifier
# linear_model.LogisticRegression


from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB

classifier_types_list = [
    KNeighborsClassifier,
    XGBClassifier,
    GradientBoostingClassifier, 
    RandomForestClassifier, 
    BaggingClassifier, 
    ExtraTreesClassifier, 
    DecisionTreeClassifier,
    LogisticRegression, 
    # Lasso, 
    RidgeClassifier,
    SVC, 
    LinearSVC, 
    # NuSVC,
    GaussianNB, 
    MultinomialNB
    ]



def get_predictions_for_the_given_classifier_type(X, y, classifier_type, n_splits, **kwargs):
    stratified_folds = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # stratified_folds.get_n_splits(X, y)
    for train_indices, test_indices in stratified_folds.split(X, y): 
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        clf = classifier_type(**kwargs)
        clf.fit(X_train,y_train)
        predictions = y.copy()
        if classifier_type == Lasso:
            cr_predictions = clf.predict(X_test)
            print("lasso: ", cr_predictions)
            cr_predictions = np.round(clf.predict(X_test))
            print("lasso - Rounded: ", cr_predictions)
            np.round()
        predictions[test_indices] = clf.predict(X_test)
    return predictions



def determine_best_performing_classifier(X, y, classifier_types_list, n_splits=12):
    f1_scores_list_for_classifiers = []
    
    for clf_type in classifier_types_list:
        print(clf_type.__name__)
        cr_predictions = get_predictions_for_the_given_classifier_type(X, y, clf_type, n_splits)
        cr_fbeta_score = fbeta_score(y, cr_predictions, beta=1.0)
        f1_scores_list_for_classifiers.append(cr_fbeta_score)
        print("cr_fbeta_score: ", cr_fbeta_score)
    
    best_performing_classifier_index = np.array(f1_scores_list_for_classifiers).argmax()
    best_performing_classifier_f1_score = f1_scores_list_for_classifiers[best_performing_classifier_index]
    best_performing_classifier_name = classifier_types_list[best_performing_classifier_index]
    print("\n\nbest_performing_classifier_name", best_performing_classifier_name, "best_performing_classifier_f1_score: ", best_performing_classifier_f1_score)



determine_best_performing_classifier(X, y, classifier_types_list)











# FEATURE SELECTION

# sorted_features_list = bfs.BiDirectionalSelection(data, 'churn', 3, RandomForestClassifier(n_jobs=-2, n_estimators=3), sequence=['f'])

# print(sorted_features_list)

sorted_features_list = ['customer service calls', 'state_DC', 'state_NM', 'voice mail plan', 'state_NE', 'state_AR', 'state_AZ', 'international plan', 'state_IA', 'state_VA', 'state_SC', 'state_CA', 'state_NJ', 'area code_408', 'state_ID', 'state_WA', 'state_UT', 'state_PA', 'state_GA', 'state_DE', 'state_LA', 'state_VT', 'state_AL', 'state_CO', 'state_NH', 'state_OH', 'state_HI', 'state_OK', 'state_CT', 'total day charge', 'total eve minutes', 'state_MD', 'state_NC', 'number vmail messages', 'total night charge', 'total day minutes', 'state_MN', 'total night minutes', 'state_MS', 'state_TN', 'state_FL', 'state_AK', 'total eve charge', 'state_OR', 'state_WV', 'state_ME', 'state_MI', 'state_NV', 'state_RI', 'state_ND', 'state_MO', 'state_KS', 'state_NY', 'state_IL', 'total intl charge', 'total night calls', 'area code_510', 'state_KY', 'state_WI', 'state_IN', 'state_SD', 'account length', 'total day calls', 'state_MA', 'state_WY', 'state_MT', 'area code_415', 'total intl minutes', 'state_TX', 'total eve calls', 'total intl calls']


# max_depths = [5,8,12,16,20,24,28,32,36]
# n_features = [10,20,30,40,50,55,60,65,69,72]

# max_depths = [20,21,22,23,24,25,26,27]
# n_features = [41,43,47,50,52,54,55]



# PARAMETER TUNING

def get_f1_score_for_random_forest_model(X, y, n_feature, max_depth, threshold, n_estimators):
    
    X = data[sorted_features_list[:n_feature]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, shuffle=True)
    
    
    
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, n_jobs=-2)
    clf.fit(X_train, y_train)
    
    # predictions = clf.predict(X_test)
    
    
    threshold = threshold
    
    prediction_probablities = clf.predict_proba(X_test)
    predictions = (prediction_probablities [:,1] >= threshold).astype('float')
    
    
    # print("\nAccuracy Score: {0}".format(accuracy_score(y_test, predictions)))
    # print("Precision Score: {0}".format(precision_score(y_test, predictions)))
    # print("Recall Score: {0}".format(recall_score(y_test, predictions)))
    cr_f1_score = fbeta_score(y_test, predictions, beta=1.0)
    print("F-Beta Score: {0}".format(cr_f1_score))
    return cr_f1_score



def get_best_performing_parameters(max_depths = [18,19,20,21], n_features = [52,53,54], thresholds = [0.41, 0.42, 0.43, 0.44], n_iterations=50):
    parameters_list = []
    f1_scores_list = []
    
    
    for max_depth in max_depths:
        for n_feature in n_features:
            for threshold in thresholds:
                cr_f1_scores_list = []
                for i in range(n_iterations):
                    
                    
                    cr_f1_score = get_f1_score_for_random_forest_model(X, y, n_feature, max_depth, threshold, n_estimators=180)
                    
                    cr_f1_scores_list.append(cr_f1_score)
                    
                parameters_list.append((max_depth, n_feature, threshold))
                f1_scores_list.append(statistics.mean(cr_f1_scores_list))
       
    
    best_parameters_tuple = parameters_list[np.array(f1_scores_list).argmax()]
    best_depth, best_n_features, best_threshold = best_parameters_tuple[0], best_parameters_tuple[1], best_parameters_tuple[2]
    print("best performing model parameters  >  max_depth: ", best_depth, " n_feature: ", best_n_features, " threshold: ", best_threshold, " F-1 Score: ", f1_scores_list[np.array(f1_scores_list).argmax()])
    
    return best_depth, best_n_features, best_threshold
    

best_depth, best_n_features, best_threshold = get_best_performing_parameters(n_iterations=20)





# print("X shapes: X: {},  y: {}".format(X.shape, y.shape))

# print("X shapes: X_up_sampled: {},  y_up_sampled: {}".format(X_up_sampled.shape, y_up_sampled.shape))


def get_mean_confusion_matrix(confusion_matrices_list):
    sum_matrix = np.zeros_like(confusion_matrices_list[0])
    for cm in confusion_matrices_list:
        sum_matrix += cm
    return sum_matrix / len(confusion_matrices_list)




def get_f1_score_for_best_parameters(best_depth, best_n_features, best_threshold, n_iterations=80):
    X_up_sampled, y_up_sampled = upsampling.up_sample_minority_class(data, 'churn', sorted_features_list[:best_n_features])
    
    X = data[sorted_features_list[:best_n_features]]
    y = data.churn
    
    
    cr_f1_scores_list = []
    cr_accuracy_scores_list = []
    cr_precision_scores_list = []
    cr_recall_scores_list = []
    
    confusion_matrices = []
    
    for i in range(n_iterations):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, shuffle=True)
        X_train_up_sampled, X_test_up_sampled, y_train_up_sampled, y_test_up_sampled = train_test_split(X_up_sampled, y_up_sampled, test_size=0.25, stratify=y_up_sampled, shuffle=True)
        
        # print("X shapes: X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        # print("X shapes_up_sampled: X_train: {}, X_test_up_sampled: {}, y_train_up_sampled: {}, y_test_up_sampled: {}".format(X_train_up_sampled.shape, X_test_up_sampled.shape, y_train_up_sampled.shape, y_test_up_sampled.shape))
        
        clf = RandomForestClassifier(max_depth=best_depth, n_estimators=80, n_jobs=-2)
        clf.fit(X_train_up_sampled, y_train_up_sampled)
        
        predictions = clf.predict(X_test)
        
        
        threshold = best_threshold
        
        prediction_probablities = clf.predict_proba(X_test)
        predictions = (prediction_probablities [:,1] >= threshold).astype('float')
        
        
        
        cr_accuracy_scores_list.append(accuracy_score(y_test, predictions))
        cr_precision_scores_list.append(precision_score(y_test, predictions))
        cr_recall_scores_list.append(recall_score(y_test, predictions))
        cr_f1_scores_list.append(fbeta_score(y_test, predictions, beta=1.0))
        
        cr_confusion_matrix = confusion_matrix(y_test, predictions)
        confusion_matrices.append(cr_confusion_matrix)
        
        if i == n_iterations - 1:
            plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
            plt.show()
            
        
    mean_confusion_matrix = get_mean_confusion_matrix(confusion_matrices)
    
    print("\nAccuracy Score: {0}".format(statistics.mean(cr_accuracy_scores_list)))
    print("Precision Score: {0}".format(statistics.mean(cr_precision_scores_list)))
    print("Recall Score: {0}".format(statistics.mean(cr_recall_scores_list)))
    print("F-Beta Score: {0}".format(statistics.mean(cr_f1_scores_list)))
    
    axis_labels = ['ongoing', 'churned']
    sns.heatmap(mean_confusion_matrix, annot=True, fmt='.2f', cmap=plt.cm.Blues, xticklabels=axis_labels, yticklabels=axis_labels)

    
    


get_f1_score_for_best_parameters(best_depth, best_n_features, best_threshold)

