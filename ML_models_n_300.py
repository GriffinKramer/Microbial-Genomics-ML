# Imports/Data Initialization
import sklearn as sk
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, balanced_accuracy_score
import openpyxl
import io

filepath = "shared-team/XX50235/Lauren/Assignment-2/"
datapath = filepath + "14-18kmerdata.txt"
metadatapath = filepath + "14-18metadata"

data = pd.read_csv(datapath, sep = "\t")
meta = pd.read_csv(metadatapath, sep=',', index_col=0)

'''
print(data.shape)
print(meta.shape)
print(meta.head())
print(meta.columns.tolist())
print(meta['Country'].value_counts())
print(meta['Region'].value_counts())
'''
################################################################

# Data Pre-processing 
data = data.set_index(data.columns[0]) #set kmer name as index
data = data.T 

common_samples = data.index.intersection(meta.index)
print("samples in data:", len(data.index))
print("samples in metadata:", len(meta.index))
print("samples in common:", len(common_samples))

data = data.loc[common_samples]
meta = meta.loc[common_samples]
'''
cutoff = 0.2 * len(data) #filter rare features 
data = data.loc[:, (data > 0).sum(axis=0) > cutoff]
print("shape after rare kmer removal:", data.shape)
data = data.div(data.sum(axis=1), axis=0) * 100 #normalize data
'''

country_counts = meta['Country'].value_counts()
valid_countries = country_counts[country_counts > 10].index
meta = meta[meta['Country'].isin(valid_countries)] #drop countries with only 1 count
data = data.loc[meta.index]
print("samples after dropping single-country samples:", len(meta))

# cap N class
n_cap = 300
n_indices = meta[meta['Country'] == 'N'].sample(n=n_cap, random_state=28).index
non_n_indices = meta[meta['Country'] != 'N'].index
meta = meta.loc[non_n_indices.append(n_indices)]
data = data.loc[meta.index]
print("samples after capping N:", len(meta))
print(meta['Country'].value_counts().head(10))
################################################################


# Train/Test Split
y_country = meta['Country'].values
y_region = meta['Region'].values

train_kmers_country, test_kmers_country, train_country, test_country = train_test_split(
    data, y_country, test_size=0.25, random_state=28, stratify=y_country)

train_kmers_region, test_kmers_region, train_region, test_region = train_test_split(
    data, y_region, test_size=0.25, random_state=28, stratify=y_region)
################################################################

# Data Reblancing

## No Rebalancing
#nothing needs to be done here

## Randomly oversample all classes
ros = RandomOverSampler(random_state=28)
train_kmers_resampled_country, train_country_resampled = ros.fit_resample(train_kmers_country.values, train_country)
train_kmers_resampled_region, train_region_resampled = ros.fit_resample(train_kmers_region.values, train_region)

print("resampled country training size:", train_kmers_resampled_country.shape)
print("resampled region training size:", train_kmers_resampled_region.shape)

# Feature selection
selector_country = SelectKBest(chi2, k=2000)
train_kmers_resampled_country = selector_country.fit_transform(train_kmers_resampled_country, train_country_resampled)
test_kmers_country = selector_country.transform(test_kmers_country.values)

selector_region = SelectKBest(chi2, k=2000)
train_kmers_resampled_region = selector_region.fit_transform(train_kmers_resampled_region, train_region_resampled)
test_kmers_region = selector_region.transform(test_kmers_region.values)

train_kmers_country = selector_country.transform(train_kmers_country.values)
train_kmers_region = selector_region.transform(train_kmers_region.values)
################################################################

# Model types (Training)
# -------------------------------------
## Gradient Boosted Decision Trees (GBDT)
### country
#### resampled
print("training GBDT country resampled...")
start = time.time()
gbdt_country_resampled = GradientBoostingClassifier(n_estimators=150, random_state=28)
gbdt_country_resampled.fit(train_kmers_resampled_country, train_country_resampled)
print("done in", round(time.time()-start, 2), "seconds")

#### no resampling
print("training GBDT country no rebalancing...")
start = time.time()
gbdt_country = GradientBoostingClassifier(n_estimators=150, random_state=28)
gbdt_country.fit(train_kmers_country, train_country)
print("done in", round(time.time()-start, 2), "seconds")

### region
#### resampled 
print("training GBDT region resampled...")
start = time.time()
gbdt_region_resampled = GradientBoostingClassifier(n_estimators=150, random_state=28)
gbdt_region_resampled.fit(train_kmers_resampled_region, train_region_resampled)
print("done in", round(time.time()-start, 2), "seconds")

#### no resampling
print("training GBDT region no rebalancing...")
start = time.time()
gbdt_region = GradientBoostingClassifier(n_estimators=150, random_state=28)
gbdt_region.fit(train_kmers_region, train_region)
print("done in", round(time.time()-start, 2), "seconds")

# -------------------------------------
## Random Forest
### country
#### resampled
print("training RF country resampled...")
start = time.time()
rf_country_resampled = RandomForestClassifier(n_estimators=10000, random_state=28, n_jobs=-1, class_weight='balanced')
rf_country_resampled.fit(train_kmers_resampled_country, train_country_resampled)
print("done in", round(time.time()-start, 2), "seconds")

##### no rebalancing
print("training RF country no rebalancing...")
start = time.time()
rf_country = RandomForestClassifier(n_estimators=10000, random_state=28, n_jobs=-1, class_weight='balanced')
rf_country.fit(train_kmers_country, train_country)
print("done in", round(time.time()-start, 2), "seconds")

### region 
#### resampled
print("training RF region resampled...")
start = time.time()
rf_region_resampled = RandomForestClassifier(n_estimators=10000, random_state=28, n_jobs=-1, class_weight='balanced')
rf_region_resampled.fit(train_kmers_resampled_region, train_region_resampled)
print("done in", round(time.time()-start, 2), "seconds")

#### no rebalancing
print("training RF region no rebalancing...")
start = time.time()
rf_region = RandomForestClassifier(n_estimators=10000, random_state=28, n_jobs=-1, class_weight='balanced')
rf_region.fit(train_kmers_region, train_region)
print("done in", round(time.time()-start, 2), "seconds")

# -------------------------------------
## SVM (Support Vector Machine)
### country
#### resampled
print("training LinearSVC country resampled...")
start = time.time()
svm_country_resampled = LinearSVC(class_weight='balanced', random_state=28, max_iter=10000)
svm_country_resampled.fit(train_kmers_resampled_country, train_country_resampled)
print("done in", round(time.time()-start, 2), "seconds")

##### no rebalancing
print("training LinearSVC country no rebalancing...")
start = time.time()
svm_country = LinearSVC(class_weight='balanced', random_state=28, max_iter=10000)
svm_country.fit(train_kmers_country, train_country)
print("done in", round(time.time()-start, 2), "seconds")

### region 
#### resampled
print("training LinearSVC region resampled...")
start = time.time()
svm_region_resampled = LinearSVC(class_weight='balanced', random_state=28, max_iter=10000)
svm_region_resampled.fit(train_kmers_resampled_region, train_region_resampled)
print("done in", round(time.time()-start, 2), "seconds")

#### no rebalancing
print("training LinearSVC region no rebalancing...")
start = time.time()
svm_region = LinearSVC(class_weight='balanced', random_state=28, max_iter=10000)
svm_region.fit(train_kmers_region, train_region)
print("done in", round(time.time()-start, 2), "seconds")
################################################################

# Model Results
## Save Classification Results
writer = pd.ExcelWriter('results/ML_n_300_classification_reports.xlsx', engine='openpyxl')

def report_to_df(test_labels, predictions, model_name):
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.index.name = 'Class'
    df['balanced_accuracy'] = balanced_accuracy_score(test_labels, predictions)
    return df

# GBDT
report_to_df(test_country, gbdt_country_resampled.predict(test_kmers_country), 'GBDT').to_excel(writer, sheet_name='GBDT_Country_ROS')
report_to_df(test_country, gbdt_country.predict(test_kmers_country), 'GBDT').to_excel(writer, sheet_name='GBDT_Country_None')
report_to_df(test_region, gbdt_region_resampled.predict(test_kmers_region), 'GBDT').to_excel(writer, sheet_name='GBDT_Region_ROS')
report_to_df(test_region, gbdt_region.predict(test_kmers_region), 'GBDT').to_excel(writer, sheet_name='GBDT_Region_None')

# RF
report_to_df(test_country, rf_country_resampled.predict(test_kmers_country), 'RF').to_excel(writer, sheet_name='RF_Country_ROS')
report_to_df(test_country, rf_country.predict(test_kmers_country), 'RF').to_excel(writer, sheet_name='RF_Country_None')
report_to_df(test_region, rf_region_resampled.predict(test_kmers_region), 'RF').to_excel(writer, sheet_name='RF_Region_ROS')
report_to_df(test_region, rf_region.predict(test_kmers_region), 'RF').to_excel(writer, sheet_name='RF_Region_None')

# LinearSVC
report_to_df(test_country, svm_country_resampled.predict(test_kmers_country), 'LinearSVC').to_excel(writer, sheet_name='SVM_Country_ROS')
report_to_df(test_country, svm_country.predict(test_kmers_country), 'LinearSVC').to_excel(writer, sheet_name='SVM_Country_None')
report_to_df(test_region, svm_region_resampled.predict(test_kmers_region), 'LinearSVC').to_excel(writer, sheet_name='SVM_Region_ROS')
report_to_df(test_region, svm_region.predict(test_kmers_region), 'LinearSVC').to_excel(writer, sheet_name='SVM_Region_None')

writer.save()
print("classification reports saved to classification_reports.xlsx")

## Summary 
results = []
def add_result(model, label, rebalancing, test_labels, predictions):
    from sklearn.metrics import precision_recall_fscore_support
    ba = balanced_accuracy_score(test_labels, predictions)
    p, r, f, _ = precision_recall_fscore_support(test_labels, predictions, average='macro', zero_division=0)
    results.append({
        'Model': model,
        'Label': label,
        'Rebalancing': rebalancing,
        'Balanced Accuracy': round(ba, 3),
        'Macro Precision': round(p, 3),
        'Macro Recall': round(r, 3),
        'Macro F1': round(f, 3)
    })

add_result('GBDT', 'Country', 'Oversample', test_country, gbdt_country_resampled.predict(test_kmers_country))
add_result('GBDT', 'Country', 'None', test_country, gbdt_country.predict(test_kmers_country))
add_result('GBDT', 'Region', 'Oversample', test_region, gbdt_region_resampled.predict(test_kmers_region))
add_result('GBDT', 'Region', 'None', test_region, gbdt_region.predict(test_kmers_region))

add_result('RF', 'Country', 'Oversample', test_country, rf_country_resampled.predict(test_kmers_country))
add_result('RF', 'Country', 'None', test_country, rf_country.predict(test_kmers_country))
add_result('RF', 'Region', 'Oversample', test_region, rf_region_resampled.predict(test_kmers_region))
add_result('RF', 'Region', 'None', test_region, rf_region.predict(test_kmers_region))

add_result('LinearSVC', 'Country', 'Oversample', test_country, svm_country_resampled.predict(test_kmers_country))
add_result('LinearSVC', 'Country', 'None', test_country, svm_country.predict(test_kmers_country))
add_result('LinearSVC', 'Region', 'Oversample', test_region, svm_region_resampled.predict(test_kmers_region))
add_result('LinearSVC', 'Region', 'None', test_region, svm_region.predict(test_kmers_region))

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv('results/ML_n_300_summary.csv', index=False)
