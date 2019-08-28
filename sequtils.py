import sys, os
import csv
import math
import numpy as np
import operator
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn import svm, metrics, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV


from keras.models import Model, Sequential, load_model
from keras.preprocessing import sequence
from keras.optimizers import *
from keras.layers import Input, RepeatVector, TimeDistributed, merge, add
from keras.layers import Dense, Dropout, Activation, Lambda, Embedding
from keras.layers import LSTM, Bidirectional, GRU, Masking
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.core import Flatten
from keras import backend as K
from keras import objectives


class CharacterTable(object):
	'''
	Given a set of characters:
	+ Encode them to a one hot integer representation
	+ Decode the one hot integer representation to their character output
	+ Decode a vector of probabilities to their character output
	'''
	def __init__(self, chars):
		self.chars = chars
		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
		self.unknown = len(self.char_indices)

	def encode(self, C, maxlen=None):
		if( maxlen==None ): maxlen=len(C) 
		X = np.zeros((maxlen, len(self.chars)+1))
		for i, c in enumerate(C):
			try: X[i, self.char_indices[c]] = 1
			except KeyError: X[i, self.unknown] = 1
		return X

	def valueencode(self, C, maxlen=None):
		if( maxlen==None ): maxlen=len(C) 
		X = np.zeros(maxlen)
		for i, c in enumerate(C):
			try: X[i]=self.char_indices[c]
			except KeyError: X[i]=self.unknown
		return X

	def decode(self, X, calc_argmax=True):
		if calc_argmax:
				X = X.argmax(axis=-1)
		return ''.join(self.indices_char[x] for x in X)

class ChemHandler() :
	idxtofp={}
	idxtosmi={}
	def __init__(self, filename):
		with open(filename, 'r') as tsv :
			data = [line.strip().split('\t') for line in tsv]
		for i, fp, smi in data :
			self.idxtofp[i] = fp[32:-7]
			self.idxtosmi[i] = smi

	def transfp(self,idx):
		return [int(i) for i in str(self.idxtofp[idx])]

	def transsmi(self,idx):
		return self.idxtosmi[idx]

	def rdkitsmi(self,idx):
		m = Chem.MolFromSmiles( self.idxtosmi[idx] )
		smi = Chem.MolToSmiles(m)
		return str(smi) 

	def transECFP(self,idx):
		m = Chem.MolFromSmiles( self.idxtosmi[idx] )
		fp = AllChem.GetMorganFingerprintAsBitVect(m, 2)
		arr = np.zeros((1,))
		DataStructs.ConvertToNumpyArray( fp, arr )
		return arr

	def transMACCS(self,idx):
		m = Chem.MolFromSmiles( self.idxtosmi[idx] )
		fp = AllChem.GetMACCSKeysFingerprint(m)
		arr = np.zeros((1,))
		DataStructs.ConvertToNumpyArray( fp, arr )
		return arr[1:]

class CCIHandler() :
	ccichem={}
	THRES = 600
	def __init__(self, filename):
		with open(filename, 'r') as tsv :
			data = [line.strip().split('\t') for line in tsv]
		for idxs, idxt, score in data :
			idxs = str(int(idxs[4:]))
			idxt = str(int(idxt[4:]))
			try : 
				self.ccichem[idxs][idxt] = score
			except KeyError :
				self.ccichem[idxs] = {}
				self.ccichem[idxs][idxt] = score
	def augment( self, idxs, labels, testidxs ) :
		newidxs = []
		newlbls = []
		for i in range(len(idxs)):
			idx = idxs[i]
			try :
				if labels[i]==0 : continue
				targetlist = self.ccichem[idx]
				for target in targetlist :
					score = int(self.ccichem[idx][target])
					if score > self.THRES :
						if target in testidxs : continue
						else :
							newidxs.append(target)
							newlbls.append(labels[i])
			except KeyError: 
				continue
		return np.append(idxs,np.array(newidxs)), np.append(labels,np.array(newlbls))


def read_input_smi( data, ctable, MAXLEN, dim=2 ) :
	filename = "./data/Bioassay/"+data+"/active.smi"
	smiles   = [smi[:-1] for smi in open(filename,'r')]
	active   = [ctable.encode( seq, MAXLEN ) for seq in smiles]

	filename = "./data/Bioassay/"+data+"/inactive.smi"
	smiles   = [smi[:-1] for smi in open(filename,'r')]
	inactive = [ctable.encode( seq, MAXLEN ) for seq in smiles]

	X = np.array( active + inactive )
	if dim==1 : X = X.reshape((len(X), np.prod(X.shape[1:])))
	Y = np.append( np.repeat(1,len(active)), np.repeat(0,len(inactive)) )
	return X, Y

def read_input_fp( data ) :
	filename = "./data/Bioassay/"+data+"/active.fingerprint"
	active   = np.loadtxt( filename )
	filename = "./data/Bioassay/"+data+"/inactive.fingerprint"
	inactive   = np.loadtxt( filename )

	X = np.vstack( (active, inactive) )
	Y = np.append( np.repeat(1,len(active)), np.repeat(0,len(inactive)) )
	return X, Y

def save_history(history_callback, key):
	his_acc = np.array( history_callback.history["acc"] )
	his_val_acc = np.array( history_callback.history["val_acc"] )
	np.savetxt("result/"+str(key)+"acc.txt",his_acc, delimiter=",")
	np.savetxt("result/"+str(key)+"val_acc.txt",his_val_acc, delimiter=",")

def save_result(desc):
	result=""
	for rslt in desc : 
		if type(rslt)!=str : result += str('%.4f'%rslt)+"\t"
		else : result +=str(rslt)+"\t"
#	print result
	filename = str(desc[8]).split()[1]
	if desc[-1].find("CV")<0: filename+="result.txt"
	elif desc[-1].find("CV_AVG")<0: filename+="CVresult.txt"
	else : filename+="CVAVGresult.txt"
	os.system("echo \""+result+"\" >> "+filename)

def print_result( y_test, y_score, desc="" ) :
	p_class = [int(y[0]>0.5) for y in y_score ]
	return print_clf_result( y_test, y_score, p_class, desc )

def print_result2( y_test, y_score, desc="" ) :
	p_class = [int(y>0.5) for y in y_score ]
	return print_clf_result( y_test, y_score, p_class, desc )

def print_clf_result( y_true, y_score, y_pred, desc="" ) :
	cm = confusion_matrix(y_true, y_pred)
	tp = cm[0][0]
	fn = cm[0][1]
	fp = cm[1][0]
	tn = cm[1][1]
	auc = roc_auc_score( y_true, y_score )
	acc = float(tp+tn)/(tp+tn+fp+fn)
	mcc = 0 if (tp+fp==0 or tp+fn==0 or tn+fp==0 or tn+fn==0) else float((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
	tpr = 0 if tp==0 else float(tp)/(tp+fn) #sensitivity
	tnr = 0 if tn==0 else float(tn)/(tn+fp) #specificity
	ppv = 0 if tp==0 else float(tp)/(tp+fp)
	npv = 0 if fn==0 else float(tn)/(tn+fn)
	f1s = 0 if tp==0 else float(2*tp)/(2*tp+fp+fn)
	save_result( [auc, acc, mcc, tpr, tnr, ppv, npv, f1s, desc] )
	return [auc, acc, mcc, tpr, tnr, ppv, npv, f1s, y_score]

def run_svm(X_train, X_test, y_train, y_test, desc) :
	lsvm = svm.LinearSVC(C=0.05)
	clf  = lsvm
	clf.fit(X_train, y_train)
	for c in clf.coef_[0]:
		print c,
	print ""
		
	return print_clf_result( y_test, clf.decision_function(X_test), clf.predict(X_test), desc)

def run_rf(X_train, X_test, y_train, y_test, desc) :
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)
	return print_clf_result( y_test, clf.predict_proba(X_test)[:,1], clf.predict(X_test), desc)

def run_ada(X_train, X_test, y_train, y_test, desc) :
	clf = AdaBoostClassifier(n_estimators=100)
	clf.fit(X_train, y_train)
	return print_clf_result( y_test, clf.predict_proba(X_test)[:,1], clf.predict(X_test), desc)

def run_gbm(X_train, X_test, y_train, y_test, desc) :
	clf = GradientBoostingClassifier(n_estimators=100)
	clf.fit(X_train, y_train)
	return print_clf_result( y_test, clf.predict_proba(X_test)[:,1], clf.predict(X_test), desc)

def run_dt(X_train, X_test, y_train, y_test, desc) :
	clf = DecisionTreeClassifier(max_depth=5)
	clf.fit(X_train, y_train)
	return print_clf_result( y_test, clf.predict_proba(X_test)[:,1], clf.predict(X_test), desc)

def run_mlp(X_train, X_test, y_train, y_test, desc) :
	clf = MLPClassifier(alpha=1)
	clf.fit(X_train, y_train)
	return print_clf_result( y_test, clf.predict_proba(X_test)[:,1], clf.predict(X_test), desc)

def run_svm2(X1_train, X2_train, X_test, y_train, y_test, desc) :
	X_train = np.concatenate((X1_train,X2_train), axis=1)
	X_trainR = np.concatenate((X2_train,X1_train), axis=1)
	clf = svm.LinearSVC()
	clf.fit(X_train, y_train)
	print_clf_result( y_train, clf.decision_function(X_train), clf.predict(X_train), desc)
	print_clf_result( y_train, clf.decision_function(X_trainR), clf.predict(X_trainR), desc)

def run_rf2(X1_train, X2_train, X_test, y_train, y_test, desc) :
	X_train = np.concatenate((X1_train,X2_train), axis=1)
	X_trainR = np.concatenate((X2_train,X1_train), axis=1)
	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	print_clf_result( y_train, clf.predict_proba(X_train)[:,1], clf.predict(X_train), desc)
	print_clf_result( y_train, clf.predict_proba(X_trainR)[:,1], clf.predict(X_trainR), desc)

def run_ada2(X1_train, X2_train, X_test, y_train, y_test, desc) :
	X_train = np.concatenate((X1_train,X2_train), axis=1)
	X_trainR = np.concatenate((X2_train,X1_train), axis=1)
	clf = AdaBoostClassifier()
	clf.fit(X_train, y_train)
	print_clf_result( y_train, clf.predict_proba(X_train)[:,1], clf.predict(X_train), desc)
	print_clf_result( y_train, clf.predict_proba(X_trainR)[:,1], clf.predict(X_trainR), desc)

def run_reg(X_train, X_test, y_train, y_test, desc) :
	reg = linear_model.Lasso(alpha = 0.5)
	reg.fit(X_train, y_train)
	prob = reg.predict(X_test)
	p_class = [int(y>0.5) for y in prob]
	return print_clf_result( y_test, prob, p_class, desc)

def average_len( sequences ) :
	seqlen = []
	for seq in sequences :
		seqlen.append( len(seq) )
	print "Average seq len:", np.mean(seqlen)


#---------------------------------------------------------------------------
# Input Handling
#---------------------------------------------------------------------------
def getdata(rtype, dataname, indim=20):
	MAXLEN=100
	charset  = "C=)(ON1234SF5l[]+6B-r#.7Hi8P9IeanAZ%YTuRLsGoKbtWgMdc0VfhUmEDypXk_"
	charset  = charset[:indim]
	ctable = CharacterTable(charset)
	chemdb = ChemHandler('./data/chem.fpsmi')

	filename = "./data/"+dataname+".tsv"
	with open(filename, 'r') as tsv : data = np.array( [ line.strip().split('\t') for line in tsv ])
	Y  = np.array([ int(int(s)>0) for s in data[:,1] ])

	if rtype=="smiles":
		XS = np.array([ ctable.encode(chemdb.transsmi(idx)) for idx in data[:,0]])
		X = sequence.pad_sequences(XS, maxlen=MAXLEN, truncating='post')
	elif rtype=="pubchem":
		X = np.array([ chemdb.transfp(idx) for idx in data[:,0]])
	elif rtype=="ECFP":
		X = np.array([ chemdb.transECFP(idx) for idx in data[:,0]])
	elif rtype=="MACCS":
		X = np.array([ chemdb.transMACCS(idx) for idx in data[:,0]])

	desc = rtype+" "+dataname+" "
	return X, Y, desc

