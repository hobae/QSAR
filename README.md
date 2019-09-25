# Comprehensive Ensemble in QSAR Prediction forDrug Discovery

## Abstract
#### Background
 Quantitative structure-activity relationship (QSAR) is a computational modeling method to reveal relationships between structural properties of chemical compounds and biological activities. QSAR modeling is essential for drug discovery, but it has many constraints. Ensemble-based machine learning approaches have been used to overcome constraints and obtain reliable predictions.
Ensemble learning builds a set of diversified models and combines them.
However, the most prevalent random forest and other ensemble approaches in QSAR prediction limit their model diversity to a single subject.
#### Description
We propose a comprehensive ensemble method that builds multi-subject diversified models and combines them through second-level meta-learning. In addition, we propose an end-to-end neural network-based individual classifier that can automatically extract sequential features from a simplified molecular-input line-entry system (SMILES). The proposed individual model did not show impressive results as a single model, but it was considered the most important predictor when combined, according to the interpretation of meta-learning. 
The proposed ensemble method consistently outperformed thirteen individual models on 19 bioassay datasets and demonstrated superiority over other ensemble approaches that are limited to a single subject.

<img src="figures/Figure1.png" width=450>

### Requirements
- sckikit-learn
- keras
- tensorflow
- pandas
- joblib
- numpy
- RDKit
- pillow


### Installation via pip
pip install pandas sklearn tensorflow keras ...

### Getting Started

- step1. CREATE CV and RESULTS/PROB DATA folders  
*Create fold numbers for CV. (e.g., CV/1, CV2, ..., CV5)

- step2. Download data to DATA folder  
*For the use of alternative data, please use sequtils.py to covert data type to one of followings: smiles, pubchem, ECFP, MACCS

- step3. run all datasets of qsar_classifiy.py (executing expe_runs.py will do)  
*This will create a learning models of SVM, RF, GBM, NNGB, and NN

- step4. run qsar_weightlearn.py for second-level learning

- step5. run qsar_result.py to get final results


### File Descriptions

(models.py)
- libraries for model training

(sequtils.py)
- libraries for input handling
(representation type can be smiles, pubchem, ECFP, and MACCS)

(expe_runs.py)
- run for all datasets of qsar_classifiy.py

(qsar_classify.py)
- first-level individual learning
	. 5-fold cross-validation for 2-nd level learning 
	. ./CV/
- Other individual methods on final 25% test set for 5 repetative tries
	. ./RESULTS/PROB/

(qsar_weightlearn.py)
- second-level for combining
	. read from CV (5-fold) and make output on ./RESULTS/PROB/. 5 repeat

(qsar_result.py)
- mean of 5 repetative tries 
- final results
