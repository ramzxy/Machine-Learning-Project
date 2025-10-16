import pandas as pd
import pandasql as ps
import numpy as np

dataset = pd.read_csv("dataset_37_diabetes.csv")

def query(sql):
    return ps.sqldf(sql, globals())



positive = dataset[dataset['class'] == 'tested_positive']
negative = dataset[dataset['class'] == 'tested_negative']

features = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
variance_analysis = []

for feature in features:
    pos_mean = positive[feature].mean()
    neg_mean = negative[feature].mean()
    diff = abs(pos_mean - neg_mean)
    pos_std = positive[feature].std()
    neg_std = negative[feature].std()
    
    variance_analysis.append({
        'feature': feature,
        'mean_diff': round(diff, 2),
        'positive_mean': round(pos_mean, 2),
        'negative_mean': round(neg_mean, 2),
        'positive_std': round(pos_std, 2),
        'negative_std': round(neg_std, 2),
        'separation_score': round(diff / ((pos_std + neg_std) / 2), 3)
    })


df_variance = pd.DataFrame(variance_analysis)
df_variance = df_variance.sort_values('separation_score', ascending=False)
print(df_variance.to_string(index=False))

print("\n\n2. FEATURE CORRELATIONS (without missing values)")
print("-" * 80)

dataset_clean = dataset.copy()
for col in ['plas', 'pres', 'skin', 'insu', 'mass']:
    dataset_clean[col] = dataset_clean[col].replace(0, np.nan)

dataset_clean['class_numeric'] = (dataset_clean['class'] == 'tested_positive').astype(int)

print("Correlation with TARGET (class):")
correlations = []
for feature in features:
    corr = dataset_clean[[feature, 'class_numeric']].corr().iloc[0, 1]
    correlations.append({'feature': feature, 'correlation': round(corr, 3)})

df_corr = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
print(df_corr.to_string(index=False))