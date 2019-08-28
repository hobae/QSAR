from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from bayes_opt import BayesianOptimization

from sequtils     import *
from models import *

dataname  =str(sys.argv[2]) if len(sys.argv)>2 else "U_1915"
MAXLEN = 100
seed  =1

#---------------------------------------------------------------------------
# Input Handling
#---------------------------------------------------------------------------
def getdata(rtype, dataname, indim=20):
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


data, target, desc = getdata("pubchem", dataname)

def svccv(C):
  val = cross_val_score(
    svm.LinearSVC(C=C),
    data, target, 'roc_auc', cv=5
  ).mean()
  return val

def rfccv(n_estimators):
  val = cross_val_score(
    RandomForestClassifier(n_estimators=int(n_estimators)),
    data, target, 'roc_auc', cv=5
  ).mean()
  return val

def adacv(n_estimators):
  val = cross_val_score(
    GradientBoostingClassifier(n_estimators=int(n_estimators)),
    data, target, 'roc_auc', cv=5
  ).mean()
  return val



def cv_smi_test(EPOCH, ffn, n_filters, filter_len, rnn_len, dr1, dr2, dr3, indim):
	X, Y, desc = getdata("smiles", dataname, int(indim))
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)
#	EPOCH=1

	params = [int(EPOCH), int(ffn), int(n_filters), int(filter_len), int(rnn_len), float(dr1), float(dr2), float(dr3), int(indim)]
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
	cvidx = 0
	result=[]
	for tr, te in kfold.split(X_train, y_train):
		cvidx+=1
		rslt = smi_model_train( X_train[tr], y_train[tr], X_train[te], y_train[te], desc+" smiles CV", params)
		result.append(rslt[0:8])

	result = np.array(result)
	avgres = [ np.mean(result[:,i]) for i in range(len(result[0])) ]
	avgres.append(desc+" CV_AVG"+str(params))
	save_result(avgres)
	print(avgres[0])
	return avgres[0]

def setbase():
	ep = 120
	fn = 32 
	fn = 128
	nf = 384
	fl = 17
	rl = 8 
	dr1 = 0.9
	dr2 = 0.4
	dr2 = 0.6
	dr3 = 0.6
	indim = 9 
	indim = 10 
	return [ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]

if __name__ == "__main__":
	gp_params = {"alpha": 1e-5}


	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for fn in [32, 64, 128, 256]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for ep in [100, 120]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for dr2 in [0.2, 0.4, 0.6]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for indim in [5, 9, 10, 20, 30]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)
	
	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for fl in [15, 17, 19, 21, 23]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for nf in [128, 192, 256, 320, 384, 512]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for rl in [8, 16, 32, 64]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for dr1 in [0.5, 0.7, 0.9]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)

	[ep,fn,nf,fl,rl,dr1,dr2,dr3,indim]=setbase()
	for dr3 in [0.4, 0.5, 0.6, 0.7, 0.8]:
		cv_smi_test(ep, fn, nf, fl, rl, dr1, dr2, dr3, indim)


	smiBO = BayesianOptimization( cv_smi_test,{
		'EPOCH': (20,150),
		'ffn': (32, 256),
		'n_filters': (32, 512),
		'filter_len': (3, 30),
		'rnn_len': (3, 30),
		'dr1':(0,1),
		'dr2':(0,1),
		'dr3':(0,1),
		'indim':(5, 50),
		})

	smiBO.maximize(n_iter=100, **gp_params)
	print('-' * 53)
	print(smiBO.res['max']['max_val'])
	print(smiBO.res['max'])
	print(smiBO.res['all'])

	svcBO = BayesianOptimization(svccv, {'C': (0.001, 100)})
	svcBO.explore({'C': [0.001, 0.05, 0.01, 0.1, 1, 10, 100]})
	svcBO.maximize(init_points=0, n_iter=1, **gp_params)
	print('-' * 53)
	print(svcBO.res['max'])

	rfcBO = BayesianOptimization( rfccv,{'n_estimators': (1, 1000)})
	rfcBO.explore({'n_estimators': [1, 10, 100, 1000]})
	rfcBO.maximize(init_points=0, n_iter=10, **gp_params)
	print('-' * 53)
	print(rfcBO.res['max'])

	adaBO = BayesianOptimization( adacv,{'n_estimators': (1, 1000)})
	adaBO.explore({'n_estimators': [1, 10, 100, 1000]})
	adaBO.maximize(init_points=0, n_iter=10, **gp_params)
	print('-' * 53)
	print(adaBO.res['max'])
