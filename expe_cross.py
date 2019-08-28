import sys
import os
import time

method = ["fp"]
dataname = ["U_1851_2c19"]
EPOCH  = [30, 50]
n_dense= [ 32, 64, 128, 256]
n_filters = [32, 64, 128, 256]
filter_len = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
rnn_len    = [4, 8, 16, 32, 64, 128]
drop_rate1  = [0.5, 0.6]
drop_rate2  = [0.1, 0.2]
dense_l1   = [256,512]
dense_l2   = [64, 128, 256]


ep=30
nl=1
dl1=512
dl2=64
dr1=0.6
dr2=0.2
data="U_1851_2c19"


for ep in EPOCH:
	for dl1 in dense_l1:
		for dl2 in dense_l2:
			for dr1 in drop_rate1:
				for dr2 in drop_rate2:
					command = "python qsar_cvfp.py fp "
					command += data+" "+str(ep)+" "+str(nl)+" "+str(dl1)+" "+str(dl2)+" "+str(dr1)+" "+str(dr2)
					print command
					os.system(command)
#=============================================================================================
method = ["fp"]
dataname = ["U_1851_2c19"]
EPOCH  = [100]
n_dense= [ 32, 64, 128, 256]
n_filters = [32, 64, 128, 256]
filter_len = [22,24,26]
rnn_len    = [ 8, 32 ]
drop_rate1  = [0.5, 0.9]
drop_rate2  = [0.2, 0.3, 0.4, 0.5]
drop_rate3  = [0.2, 0.5]

for ep in EPOCH:
	for dl1 in dense_l1:
		for dl2 in dense_l2:
			for dr1 in drop_rate1:
				for dr2 in drop_rate2:
					command = "python qsar_cvsmi.py fp "
					command += data+" "+str(ep)+" "+str(nl)+" "+str(dl1)+" "+str(dl2)+" "+str(dr1)+" "+str(dr2)
					print command
					os.system(command)



ep=100
nd=128
nf=256
fl=22
rl=8
dr1=0.9
dr2=0.4
dr3=0.5

for nd in n_dense:
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
nd=128

for nf in n_filters :
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
nf=256

for fl in filter_len:
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
fl=22

for rl in rnn_len:
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
rl=8

for dr1 in drop_rate1:
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
dr1=0.9

for dr2 in drop_rate2:
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
dr2=0.4

for dr3 in drop_rate3:
	command = "python qsar_cvsmi.py smi "
	command += dataname[0]+" "+str(ep)+" "+str(nd)+" "+str(nf)+" "+str(fl)+" "+str(rl)+" "+str(dr1)+" "+str(dr2)+" "+str(dr3)
	print command
	os.system(command)
dr3=0.5


