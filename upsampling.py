import pandas as pd
import numpy as np
from sklearn.utils import resample

def up_sample_minority_class(data, target_feature, feature_filter_list=[]):
    # Entries of the both minority and majority classes
    value_majority = data[target_feature].value_counts().sort_values(ascending=False).index[0]
    data_majority = data.loc[data[target_feature] == value_majority]
    data_minority = data.loc[data[target_feature] != value_majority]
    
    print("data_majority: {0} @ data_minority: {1}".format(len(data_majority), len(data_minority)))
    
    #populates the minority portion of the samples up to the size of majority portion
    data_minority_up_sampled = resample(data_minority, 
                                     replace=True,
                                     n_samples=len(data_majority),
                                     random_state=142)
    
    # Combine majority class with upsampled minority class
    data_up_sampled = pd.concat([data_majority, data_minority_up_sampled])
    
    # Display new class counts
    print(data_up_sampled[target_feature].value_counts())
    
    
    if len(feature_filter_list) == 0:
        X_up_sampled = np.array(data_up_sampled.drop([target_feature], 1).astype(float))
    else:
        X_up_sampled = np.array(data_up_sampled[feature_filter_list].astype(float))
    
    
    y_up_sampled = np.array(data_up_sampled[target_feature]).astype(float)
    
    
    # print("X_up_sampled: ",  len(X_up_sampled), "  y_up_sampled: ",  len(y_up_sampled))
    
    return X_up_sampled, y_up_sampled
