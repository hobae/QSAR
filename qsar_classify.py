from sequtils import *
from models import *
import os

GPUID = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

#---------------------------------------------------------------------------
# Parameters
#---------------------------------------------------------------------------
MAXLEN = 100
method    =str(sys.argv[1]) if len(sys.argv)>1 else "fp"
dataname  =str(sys.argv[2]) if len(sys.argv)>2 else "U_1851_2c19"
EPOCH     =int(sys.argv[3]) if len(sys.argv)>3 else 120
FFL       =int(sys.argv[4]) if len(sys.argv)>4 else 32
n_filters =int(sys.argv[5]) if len(sys.argv)>5 else 384
filter_len=int(sys.argv[6]) if len(sys.argv)>6 else 17
rnn_len   =int(sys.argv[7]) if len(sys.argv)>7 else 8
dr1       =float(sys.argv[8]) if len(sys.argv)>8 else 0.9
dr2       =float(sys.argv[9]) if len(sys.argv)>9 else 0.6
dr3       =float(sys.argv[10]) if len(sys.argv)>10 else 0.6
indim     =int(sys.argv[11]) if len(sys.argv)>11 else 9
isRandom  =False #otherwise define split option such as StratifiedKFold, TimeSeriesSplit


#---------------------------------------------------------------------------
# Final Test
#---------------------------------------------------------------------------
def final_test(idx, X_train, y_train, X_test, y_test, repre, method, desc, path ) :
	print(idx)
	MAXITR=100
	desc = desc+" "+method
	fp  = open(path+str(idx)+"/"+dataname+"_"+repre+"_"+method+"_pred.txt",'w')

	if repre=="smiles" :
		params = [EPOCH, FFL, n_filters, filter_len, rnn_len, dr1, dr2, dr3, indim]
		y_p = smi_model_train( X_train, y_train, X_test, y_test, desc, params)[8][:,0]
	else :
		if method.startswith("svm")   :y_p = run_svm ( X_train, X_test, y_train, y_test, desc)[8]
		elif method.startswith("rf")  :y_p = run_rf  ( X_train, X_test, y_train, y_test, desc)[8]
		elif method.startswith("gbm") :y_p = run_gbm ( X_train, X_test, y_train, y_test, desc)[8]
		elif method.startswith("nnbg"):y_p = fp_model_train_bagging( X_train, y_train, X_test, y_test, MAXITR, desc)[8][:,0]
		elif method.startswith("nn")  :y_p = fp_model_train( X_train, y_train, X_test, y_test, desc)[8][:,0]
		else : y_p=y_test

	for i in range(len(X_test)):
		fp.write(str(y_p[i])+"\n")
	fp.close()
	# cross validation roop


#---------------------------------------------------------------------------
# Input Handling
#--------------------------------------------------------------------------
repres = ["smiles", "pubchem", "ECFP", "MACCS"]
methods= ["nn","svm0.05","rf100","gbm100"]
seed  = 1


#------------- CV for model selection and for weight learning ----------------------------------
#------------- First-level individual learning in CV -------------------------------------------

if (isRandom):
    X, Y, desc = getdata("pubchem", dataname, indim)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)
    final_test(0, X_train, y_train, X_train, y_train, "label", "", desc, "./CV/") 

else:
    kfold = TimeSeriesSplit(n_splits=5)
    cvidx = 0
    result=[]
    X_train, y_train, desc = getdata("pubchem", dataname, indim)
    for tr, te in kfold.split(X_train, y_train):
        print (len(tr))
        cvidx+=1
        final_test(cvidx, X_train[tr], y_train[tr], X_train[te], y_train[te], "label", "", desc, "./CV/") 
        for repre in repres:
            X_train, y_train, desc = getdata(repre, dataname, indim)
            if repre=="smiles": final_test(cvidx, X_train[tr], y_train[tr], X_train[te], y_train[te], repre, methods[0], desc, "./CV/") 
            else:
                for method in methods: final_test(cvidx, X_train[tr], y_train[tr], X_train[te], y_train[te], repre, method, desc, "./CV/") 


#-------------- Five repetative test (Conventional & Plain NN for compair) ---------------------
if (isRandom):
    X, Y, desc = getdata("pubchem", dataname, indim)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)

    for i in range(0,21,1):
        print("repeat : %s" % i)
        final_test(i, X_train, y_train, X_test, y_test, "label", "", desc, "./RESULTS/PROB/") 
        for repre in repres:
            X, Y, desc = getdata(repre, dataname, indim)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)
            if repre=="smiles": final_test(i, X_train, y_train, X_test, y_test, repre, methods[0], desc, "./RESULTS/PROB/") 
            else:
                for method in methods: final_test(i, X_train, y_train, X_test, y_test, repre, method, desc, "./RESULTS/PROB/") 

else:
    print ("Five repetative test")
    X, Y, desc = getdata("pubchem", dataname, indim)
    kfold = TimeSeriesSplit(n_splits=5)
    cvidx = 0
    result=[]
    X_train, y_train, desc = getdata("pubchem", dataname, indim)

    for tr, te in kfold.split(X_train, y_train):
        cvidx+=1
        final_test(cvidx, X_train[tr], y_train[tr], X_train[te], y_train[te], "label", "", desc, "./RESULTS/PROB/") 
        for repre in repres:
            X_train, y_train, desc = getdata(repre, dataname, indim)
            if repre=="smiles": final_test(cvidx, X_train[tr], y_train[tr], X_train[te], y_train[te], repre, methods[0], desc, "./RESULTS/PROB/") 
            else:
                for method in methods: final_test(cvidx, X_train[tr], y_train[tr], X_train[te], y_train[te], repre, method, desc, "./RESULTS/PROB/") 

