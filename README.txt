(directory)
working directory: /HOME/USERS/QSAR/
CV/ : 5-fold cross-validation
RESULTS/PROB/ : 5 independent repeat tests

(models.py), (sequtils.py)
- libraries

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
