
import random
import pandas as pd
from sklearn.utils import resample

class Balancer:

    def __init__(self, method='undersample', random_state=42):
        
        self.method = method
        self.random_state = random_state

    def undersample_to_minority(self, df, class_0, class_1):

        # Calculate the minority class size within each group
        class_0_min_size = min(df[df['type'].isin(class_0)]['type'].value_counts())
        class_1_min_size = min(df[df['type'].isin(class_1)]['type'].value_counts())
        
        # Determine the total number of samples needed for each group to match
        total_samples_per_group = min(class_0_min_size * len(class_0), class_1_min_size * len(class_1))
        
        # Calculate how many samples per class are needed to equally distribute
        # 'total_samples_per_group' across each class in both groups
        samples_per_class_class_0 = total_samples_per_group // len(class_0)
        samples_per_class_class_1 = total_samples_per_group // len(class_1)
        
        # Undersample data for each class in both groups for binary classification
        undersampled_class_0 = pd.concat([df[df['type'] == cls].sample(n=samples_per_class_class_0,
                                                                       random_state=self.random_state) for cls in class_0])
        
        undersampled_class_1 = pd.concat([df[df['type'] == cls].sample(n=samples_per_class_class_1,
                                                                       random_state=self.random_state) for cls in class_1])
        
        # Combine the undersampled groups
        undersampled_df = pd.concat([undersampled_class_0, undersampled_class_1]).reset_index(drop=True)
        
        return undersampled_df
        
    def oversample_to_majority(self, df, class_0, class_1):

        # Calculate the majority class size within each group
        class_0_maj_size = max(df[df['type'].isin(class_0)]['type'].value_counts())
        class_1_maj_size = max(df[df['type'].isin(class_1)]['type'].value_counts())
        
        # Determine the total number of samples needed for each group to match
        total_samples_per_group = max(class_0_maj_size * len(class_0), class_1_maj_size * len(class_1))
        
        # Calculate how many samples per class are needed to equally distribute
        # 'total_samples_per_group' across each class in both groups
        samples_per_class_class_0 = total_samples_per_group // len(class_0)
        samples_per_class_class_1 = total_samples_per_group // len(class_1)
        
        # Oversample data for each class in both groups for binary classification
        oversampled_class_0 = pd.concat([resample(df[df['type'] == cls], replace=True, n_samples=samples_per_class_class_0,
                                                  random_state=self.random_state) for cls in class_0])
        
        oversampled_class_1 = pd.concat([resample(df[df['type'] == cls], replace=True, n_samples=samples_per_class_class_1,
                                                  random_state=self.random_state) for cls in class_1])
        
        # Combine the oversampled groups
        oversampled_df = pd.concat([oversampled_class_0, oversampled_class_1]).reset_index(drop=True)
        
        return oversampled_df
    
    def apply(self, df, tree_pair_dict):

        balanced_results = {}
        for stage, classes in tree_pair_dict.items():
            
            if self.method=='undersample':
                balanced_df = self.undersample_to_minority(df, classes['class_0'], classes['class_1'])
            elif self.method=='oversample':
                balanced_df = self.oversample_to_majority(df, classes['class_0'], classes['class_1'])

            balanced_results[stage] = balanced_df
            
        return balanced_results