import sys
import os
import time
import re
import math
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from sequtils import *

def readdata(path, cvidx, dataname, repre, method):
	filename = path+str(cvidx)+"/"+dataname+"_"+repre+"_"+method+"_pred.txt"
	pr = np.loadtxt(filename)
	return pr.tolist()

def aggregation(tpr):
	pr = np.mean((tpr), axis=0)
	return pr

#===============================================================================

EVL_MTR = 0  # AUC
datanames = [  "U_1851_1a2", "U_1851_2c19", "U_1851_2c9", "U_1851_2d6", "U_1851_3a4",
              "U_1915", "U_2358", "U_463213", "U_463215", "U_488912", "U_488915",
              "U_488917", "U_488918", "U_492992", "U_504607", "U_624504",
							"U_651739",
              "U_651744", "U_652065"]


path_cv = "./CV/"
path_te = "./RESULTS/PROB/"
#--------------------------------------
# data reading
#--------------------------------------
repres = ["pubchem", "ECFP", "MACCS", "smiles"]
methods= ["rf100", "svm0.05","gbm100","nn"]


def readdataall(path, idx, dataname, repres, methods):
	total_pr=[]
	for repre in repres :
		for method  in methods :
			if repre=="smiles" and method!="nn" : continue
			if repre=="smiles" : continue
		if repre!="smiles" and method=="nn" : continue
			total_pr.append(readdata(path, idx, dataname, repre, method))
	total_pr = np.transpose(np.array(total_pr))
	return total_pr.tolist()

def result_write(filename, y_p):
	fp = open(filename,"w")
	for y in y_p:
		fp.write(str(y)+"\n")
	fp.close()

def bayesian_optimal(x):
	ps_count=0.001
	y_pred=[]
	for i in range(x.shape[0]):
		score=1
		for j in range(x.shape[1]):
			score*=(x[i][j]+ps_count)
			score/=(1-x[i][j]+ps_count)
		y_pred.append(score)
	return y_pred


for dataname in datanames:

	x_train = []
	y_train = []
	for cv in range(1,6):
		x_train.extend(readdataall(path_cv, cv, dataname, repres, methods))
		y_train.extend(readdata(path_cv, cv, dataname, "label",""))

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	for cv in range(1,6):
		x_test = []
		y_test = []
		x_test=readdataall(path_te, cv, dataname, repres, methods)
		y_test=readdata(path_te, cv, dataname, "label","")
		x_test = np.array(x_test)
		y_test = np.array(y_test)

		print("Train shape:", x_train.shape)
		print("Test shape :", x_test.shape)
		desc = dataname+" aggre "

		#---------------------------------
		# Weight Learn by Nueral Network
		#---------------------------------
		inputs = Input(shape=(x_train.shape[1:]))
		X = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(inputs)
		X = Dropout(0.5)(X)
		X = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(X)
		X = Dropout(0.5)(X)
		X = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(X)
		X = Dropout(0.5)(X)
		Y = Dense(1, activation='sigmoid', kernel_initializer='uniform')(X)

		model = Model(inputs=inputs, outputs=Y)
		model.summary()
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		his = model.fit(x_train, y_train, nb_epoch=30, batch_size=256, shuffle=True,
										validation_data=(x_test, y_test) )
		y_pred = model.predict(x_test)
		rslt = print_result(y_test, y_pred, desc+"learning")
		y_p  = rslt[8][:,0]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_nnnn2_pred.txt", y_p)

		#---------------------------------
		# Weight Learn by Machien Learning 
		#---------------------------------
		y_p = run_svm(x_train, x_test, y_train, y_test, "aggre nnsvm")[8]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_nnsvm_pred.txt", y_p)

		y_p = run_rf(x_train, x_test, y_train, y_test, "aggre nnrf")[8]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_nnrf_pred.txt", y_p)

		y_p = run_gbm(x_train, x_test, y_train, y_test, "aggre nngbm")[8]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_nngbm_pred.txt", y_p)

		y_p = run_reg(x_train, x_test, y_train, y_test, "aggre nnreg")[8]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_nnreg_pred.txt", y_p)

		y_p = run_ada(x_train, x_test, y_train, y_test, "aggre nnreg")[8]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_nnlasso_pred.txt", y_p)

		#---------------------------------
		# Weight Learn by bayesian optimal 
		#---------------------------------
		y_pred=bayesian_optimal(x_test)
		rslt = print_result2(y_test, y_pred, desc+"learning")
		y_p  = rslt[8]
		result_write("./RESULTS/PROB/"+str(cv)+"/"+dataname+"_learned_bayes_pred.txt", y_p)


