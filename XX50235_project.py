#https://scikit-learn.org/stable/supervised_learning.html
#skikit learn

import sklearn as sk
import pandas as pd
import numpy as np

filepath = "shared-team/XX50235/Lauren/Assignment-2/"
datapath = filepath + "14-18kmerdata.txt"
metadatapath = filepath + "14-18metadata"

data = pd.read_csv(datapath, sep = "\t")
meta = pd.read_csv(metadatapath, sep='\t', compression='zip').set_index('SRA.Accession', drop = False)


#Data Pre-processing 
## No Rebalancing

## Class weight

## Downsample major class

## Upsample all classes

## Randomly oversample all classes


#Model types
# Gradient Boosted Decision Trees (GBDT)

# Random Forest

# SVM (Support Vector Machine)? 

# Artificial Neural Network?


# Feature selection
# Randomly subset

# fischer

# random forest to select