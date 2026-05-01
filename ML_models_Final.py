# Imports
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import openpyxl
import os

filepath = "shared-team/XX50235/Lauren/Assignment-2/"
data = pd.read_csv(filepath + "14-18kmerdata.txt", sep="\t")
meta = pd.read_csv(filepath + "14-18metadata", sep=',', index_col=0)
test_data_raw = pd.read_csv(filepath + "19kmerdata.txt", sep="\t")
test_meta = pd.read_csv(filepath + "19metadata", sep=',', index_col=0)

# data preprocessing 
## training data
data = data.T
common_samples = data.index.intersection(meta.index)
data = data.loc[common_samples]
meta = meta.loc[common_samples]

country_counts = meta['Country'].value_counts()
valid_countries = country_counts[country_counts > 10].index
meta = meta[meta['Country'].isin(valid_countries)]
data = data.loc[meta.index]

n_indices = meta[meta['Country'] == 'N'].sample(n=300, random_state=28).index
non_n_indices = meta[meta['Country'] != 'N'].index
meta = meta.loc[non_n_indices.append(n_indices)]
data = data.loc[meta.index]

y_country = meta['Country'].values
y_region = meta['Region'].values
data = data.astype(np.float32)

## test data
test_data = test_data_raw.T
common_test = test_data.index.intersection(test_meta.index)
test_data = test_data.loc[common_test]
test_meta = test_meta.loc[common_test]

train_cols = data.columns.tolist()
test_col_lookup = {c: i for i, c in enumerate(test_data.columns.tolist())}
test_array = np.zeros((len(test_data), len(train_cols)), dtype=np.float32)
for j, col in enumerate(train_cols):
    if col in test_col_lookup:
        test_array[:, j] = test_data.iloc[:, test_col_lookup[col]].values
test_data = pd.DataFrame(test_array, index=test_data.index, columns=train_cols)

test_meta_country = test_meta[test_meta['Country'].isin(valid_countries)]
test_data_country = test_data.loc[test_meta_country.index]
y_test_country = test_meta_country['Country'].values

y_test_region = test_meta['Region'].values
test_data_region = test_data

# train test split
train_kmers_country, trash1, train_country, trash2 = train_test_split(
    data, y_country, test_size=0.25, random_state=28, stratify=y_country)

train_kmers_region, trash1, train_region, trash2 = train_test_split(
    data, y_region, test_size=0.25, random_state=28, stratify=y_region)

# resampling
ros = RandomOverSampler(random_state=28)
X_resampled_country, y_resampled_country = ros.fit_resample(train_kmers_country.values, train_country)
X_resampled_region, y_resampled_region = ros.fit_resample(train_kmers_region.values, train_region)

# feature selection
selector_country = SelectKBest(chi2, k=10000)
X_train_country = selector_country.fit_transform(X_resampled_country, y_resampled_country)
X_test_country = selector_country.transform(test_data_country.values)

selector_region = SelectKBest(chi2, k=10000)
X_train_region = selector_region.fit_transform(X_resampled_region, y_resampled_region)
X_test_region = selector_region.transform(test_data_region.values)

# Model training 
print("training RF country...")
start = time.time()
rf_country = RandomForestClassifier(n_estimators=10000, random_state=28, n_jobs=-1, class_weight='balanced')
rf_country.fit(X_train_country, y_resampled_country)
print("done in", round(time.time()-start, 2), "seconds")

print("training RF region...")
start = time.time()
rf_region = RandomForestClassifier(n_estimators=10000, random_state=28, n_jobs=-1, class_weight='balanced')
rf_region.fit(X_train_region, y_resampled_region)
print("done in", round(time.time()-start, 2), "seconds")

# Model evalutation
pred_country = rf_country.predict(X_test_country)
pred_region = rf_region.predict(X_test_region)

print("\nRF Country:")
print(classification_report(y_test_country, pred_country, zero_division=0))
print("\nRF Region:")
print(classification_report(y_test_region, pred_region, zero_division=0))

def report_to_df(test_labels, predictions):
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.index.name = 'Class'
    df['balanced_accuracy'] = balanced_accuracy_score(test_labels, predictions)
    return df

writer = pd.ExcelWriter('results/ML_Final_classification_reports.xlsx', engine='openpyxl')
report_to_df(y_test_country, pred_country).to_excel(writer, sheet_name='RF_Country_ROS')
report_to_df(y_test_region, pred_region).to_excel(writer, sheet_name='RF_Region_ROS')
writer.save()

results = []
for label, test_labels, predictions in [
    ('Country', y_test_country, pred_country),
    ('Region',  y_test_region,  pred_region)]:
    ba = balanced_accuracy_score(test_labels, predictions)
    p, r, f, _ = precision_recall_fscore_support(test_labels, predictions, average='macro', zero_division=0)
    results.append({'Model': 'RF', 'Label': label, 'Rebalancing': 'Oversample',
                    'Balanced Accuracy': round(ba, 3), 'Macro Precision': round(p, 3),
                    'Macro Recall': round(r, 3), 'Macro F1': round(f, 3)})

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv('results/ML_Final_summary.csv', index=False)

# Confusion matrix figure
for label, test_labels, predictions in [
    ('Country', y_test_country, pred_country),
    ('Region',  y_test_region,  pred_region)]:
    classes = sorted(set(test_labels))
    cm = confusion_matrix(test_labels, predictions, labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title(f'Confusion Matrix: {label} Prediction (2019 Test Set)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{label.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    
# Feature importance table
# Feature importance Excel tables (TOP 10 ONLY)
with pd.ExcelWriter('results/ML_Final_feature_importance.xlsx', engine='openpyxl') as writer:
    for label, model, selector in [
        ('Country', rf_country, selector_country),
        ('Region',  rf_region,  selector_region)
    ]:
        kmer_names = np.array(data.columns)[selector.get_support(indices=True)]
        importances = model.feature_importances_

        # Get top 10
        top_n = 10
        top_indices = np.argsort(importances)[::-1][:top_n]

        top_df = pd.DataFrame({
            'Rank': range(1, top_n + 1),
            'Kmer': kmer_names[top_indices],
            'Mean_Decrease_in_Impurity': importances[top_indices]
        })

        # Sort properly by importance (just to be safe)
        top_df = top_df.sort_values(
            by='Mean_Decrease_in_Impurity',
            ascending=False
        ).reset_index(drop=True)

        top_df['Rank'] = range(1, top_n + 1)

        top_df.to_excel(
            writer,
            sheet_name=f'RF_{label}_Top10_Features',
            index=False
        )
        