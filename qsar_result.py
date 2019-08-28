import sys
import os
import time
import re
import math
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def readdata(cvidx, dataname, repre, method):
	filename = "./RESULTS/PROB/"+str(cvidx)+"/"+dataname+"_"+repre+"_"+method+"_pred.txt"
	pr = np.loadtxt(filename)
	return pr

def readdata2(cvidx, dataname, repre, method):
	filename = "./RESULTS/PROB/"+str(cvidx)+"/"+dataname+"_"+repre+"_pred.txt"
	pr = np.loadtxt(filename)
	return pr

def aggregation(tpr):
	pr = np.mean((tpr), axis=0)
	return pr

def get_clf_result( y_true, y_score, desc="" ) :
	y_pred = [int(y>0.5) for y in y_score ]
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
	print "\nauc    acc    mcc    tpr    tnr    ppv    npv    f1s\t"+desc
        print '%.4f'%auc, '%.4f'%acc, '%.4f'%mcc, '%.4f'%tpr, '%.4f'%tnr, '%.4f'%ppv, '%.4f'%npv, '%.4f'%f1s

	return [auc, acc, mcc, tpr, tnr, ppv, npv, f1s, y_score, mcc]

#===============================================================================

EVL_MTR = 0  # AUC
EVL_MCC = 9  # MCC
datanames = [  "U_1851_1a2", "U_1851_2c19", "U_1851_2c9", "U_1851_2d6", "U_1851_3a4",
              "U_1915", "U_2358", "U_463213", "U_463215", "U_488912", "U_488915",
              "U_488917", "U_488918", "U_492992", "U_504607", "U_624504", "U_651739",
              "U_651744", "U_652065"]

#--------------------------------------
# indivisual methods result
#--------------------------------------
#repres = ["smiles", "pubchem", "ECFP", "MACCS"]
#methods= ["nn","nnbg100-im2","svm0.05","rf100","gbm100"]
#methods= ["nn","nnbg100","svm0.05","rf100","gbm100"]
#methods= ["nnre50", "nnre100", "nnbg10", "nnbg50", "nnbg100"]
#methods= ["nn","svm0.05","rf100","gbm100"]

repres = ["smiles"]
methods= ["nn"]

for repre in repres :
	for method  in methods :
		if repre=="smiles" and method!="nn" : continue
		print(repre, method,"\n")
		for dataname in datanames :
			result=[]
			result_mcc=[]
			for cv in range(1,5):
				gt = readdata(cv, dataname, "label","")
				pr = readdata(cv, dataname, repre, method)
				result.append( get_clf_result(gt,pr)[EVL_MTR] )
				result_mcc.append( get_clf_result(gt,pr)[EVL_MCC] )
			print np.mean(result)
			print np.mean(result_mcc)


#--------------------------------------
# method ensemble
#--------------------------------------
repres = ["pubchem", "ECFP", "MACCS"]
methods= ["nn","svm0.05","rf100","gbm100"]
#methods= ["nnbg100","svm0.05","rf100","gbm100"]
for repre in repres :
	print(repre,"\n")
	for dataname in datanames :
		result=[]
		for cv in range(1,21):
			gt = readdata(cv, dataname, "label","")
			tpr=[]
			for method in methods:
				tpr.append(readdata(cv, dataname, repre, method))
			pr = aggregation(tpr)
			result.append( get_clf_result(gt,pr)[EVL_MTR] )
		print np.mean(result)
               


#--------------------------------------
# representation ensemble
#--------------------------------------
repres = ["smiles", "pubchem", "ECFP", "MACCS"]
methods= ["nn","svm0.05","rf100","gbm100"]
#methods= ["nnbg100","svm0.05","rf100","gbm100"]

for method in methods:
	print(method,"\n")
	for dataname in datanames :
		result=[]
		for cv in range(1,21):
			gt = readdata(cv, dataname, "label","")
			tpr=[]
			for repre in repres :
				tpr.append(readdata(cv, dataname, repre, method))
			pr = aggregation(tpr)
			result.append( get_clf_result(gt,pr)[EVL_MTR] )
		print np.mean(result)
#--------------------------------------
# total ensemble
#--------------------------------------
#repres = ["smiles", "pubchem", "ECFP", "MACCS"]
repres = ["smiles", "pubchem", "ECFP", "MACCS"]
methods= ["nn","svm0.05","rf100","gbm100"]
print("total","\n")
for dataname in datanames :
	result=[]
	for cv in range(1,21):
		gt = readdata(cv, dataname, "label","")
		tpr=[]
		for method in methods:
			for repre in repres :
				#if repre=="smiles" : continue
				if repre=="smiles" and method!="nn" : continue
				tpr.append(readdata(cv, dataname, repre, method))
		pr = aggregation(tpr)
		result.append( get_clf_result(gt,pr)[EVL_MTR] )
	print np.mean(result)

#--------------------------------------
# weight learned ensemble
#--------------------------------------
print("weight learned","\n")
for dataname in datanames :
	result=[]
	for cv in range(1,6):
		gt = readdata(cv, dataname, "label","")
#		pr = readdata(cv, dataname, "learned","svmtmp")
#		pr = readdata(cv, dataname, "learned","nnlasso")
#	        pr = readdata(cv, dataname, "learned","bayes")
#		pr = readdata(cv, dataname, "learned","nnnn")
		pr = readdata(cv, dataname, "learned","nnsvm")
#		pr = readdata(cv, dataname, "learned","reg")
#		pr = readdata(cv, dataname, "learned","gbm")
#		pr = readdata(cv, dataname, "learned","rf")
#		pr = readdata(cv, dataname, "learned","svm")
#		pr = readdata(cv, dataname, "learned","nns")
#		pr = readdata2(cv, dataname, "learned","")
		result.append( get_clf_result(gt,pr)[EVL_MTR] )
	print np.mean(result)
