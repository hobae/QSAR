from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sequtils     import *

#---------------------------------------------------------------------------
# Parameters
#---------------------------------------------------------------------------
MAXLEN = 100
method    =str(sys.argv[1]) if len(sys.argv)>1 else "smi"
dataname  =str(sys.argv[2]) if len(sys.argv)>2 else "U_1851_2c19"
#dataname  =str(sys.argv[2]) if len(sys.argv)>2 else "U_1851_1a2"

#---------------------------------------------------------------------------
# Input Handling
#---------------------------------------------------------------------------
charset  = "C=)(ON1234SF5l[]+6B-r#.7Hi8P9IeanAZ%YTuRLsGoKbtWgMdc0VfhUmEDypXk_"
ctable = CharacterTable(charset)
chemdb = ChemHandler('./data/chem.fpsmi')

filename = "./data/"+dataname+".tsv"
with open(filename, 'r') as tsv : data = np.array( [ line.strip().split('\t') for line in tsv ])

seed = 1
XF = np.array([ chemdb.transfp(idx) for idx in data[:,0]])
Y  = np.array([ int(int(s)>0) for s in data[:,1] ])
X_train, X_test, y_train, y_test = train_test_split(XF, Y, test_size=0.25, random_state=seed)


# Set the parameters by cross-validation
# for SVM
tuned_parameters = [{'C': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]}]
clf = GridSearchCV(svm.LinearSVC(C=1), tuned_parameters, cv=5, scoring='roc_auc')

tuned_parameters = [{'n_estimators': [1, 10, 100, 1000]}]
for RF
tuned_parameters = [{'n_estimators': [1, 10, 100, 1000, 1500, 2000, 5000]}]
clf = GridSearchCV(RandomForestClassifier(n_estimators=1), tuned_parameters, cv=5, scoring='roc_auc')

for GBM
tuned_parameters = [{'n_estimators': [1, 10, 100, 1000, 1500, 2000, 5000]}]
clf = GridSearchCV(GradientBoostingClassifier(n_estimators=1), tuned_parameters, cv=5, scoring='roc_auc')

clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
					% (mean, std * 2, params))

print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

#---------------------------------------------------------------------------
# Results
#---------------------------------------------------------------------------
Best parameters set found on development set:
{'C': 0.05}
Grid scores on development set:
0.816 (+/-0.012) for {'C': 0.0001}
0.842 (+/-0.012) for {'C': 0.001}
0.853 (+/-0.013) for {'C': 0.01}
0.853 (+/-0.014) for {'C': 0.05}
0.852 (+/-0.014) for {'C': 0.1}
0.849 (+/-0.016) for {'C': 0.5}
0.848 (+/-0.017) for {'C': 1}
0.819 (+/-0.023) for {'C': 10}
0.795 (+/-0.034) for {'C': 100}
0.795 (+/-0.034) for {'C': 1000}

Best parameters set found on development set:
{'n_estimators': 1000}
Grid scores on development set:
0.685 (+/-0.034) for {'n_estimators': 1}
0.836 (+/-0.008) for {'n_estimators': 10}
0.860 (+/-0.008) for {'n_estimators': 100}
0.864 (+/-0.006) for {'n_estimators': 1000}
0.863 (+/-0.006) for {'n_estimators': 1500}
0.863 (+/-0.005) for {'n_estimators': 2000}
0.863 (+/-0.006) for {'n_estimators': 5000}

Best parameters set found on development set:
{'n_estimators': 1000}
Grid scores on development set:
0.738 (+/-0.031) for {'n_estimators': 1}
0.791 (+/-0.022) for {'n_estimators': 10}
0.846 (+/-0.014) for {'n_estimators': 100}
0.871 (+/-0.004) for {'n_estimators': 1000}
0.871 (+/-0.004) for {'n_estimators': 1500}
0.870 (+/-0.004) for {'n_estimators': 2000}
0.863 (+/-0.003) for {'n_estimators': 5000}
