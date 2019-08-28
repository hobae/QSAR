from sequtils     import *
from models import *

#---------------------------------------------------------------------------
# Model Building
#---------------------------------------------------------------------------

def bootstrap_sampling_before( data, indices ):
  NP = len(indices[0])
  NN = len(indices[1])
  print NP, NN
  if NP < NN : SN = NP
  else : SN = NN

  pidx = np.random.choice(indices[0], SN)
  nidx = np.random.choice(indices[1], SN)
  btdata  = []
  btlabel = []
  for i in range(SN) :
    btdata += [data[pidx[i]]]
    btlabel+= [1]
    btdata += [data[nidx[i]]]
    btlabel+= [0]
  return np.array(btdata), np.array(btlabel)

def bootstrap_sampling( data, label):
	N = len(data)
	idx = np.arange(N)
	btidx = np.random.choice(idx, N)

	btdata  = []
	btlabel = []
	for i in btidx :
		btdata += [data[i]]
		btlabel+= [label[i]]
	return np.array(btdata), np.array(btlabel)


opt = Adam(lr=1e-4)

def Varweightfit( X1_train, y_train, X1_test, y_test, desc, EPOCH, model ) :
	nsamples=y_train.shape[0]
	indices = [np.where(y_train==1)[0]]
	indices += [np.where(y_train==0)[0]]
	train, label = bootstrap_sampling( X1_train, y_train)
	his = model.fit(train, label, epochs=EPOCH, batch_size=256,
							validation_data=(X1_test, y_test), verbose=2)
	return model

def smi_model_train( X_train, y_train, X_test, y_test, desc, params ) :
	EPOCH, FFL, n_filters, filter_len, rnn_len, dr1, dr2, dr3, indim= params
	desc += str(params)

	print('SMI Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	inputs = Input(shape=(X_train.shape[1:]))

	X = Convolution1D(n_filters,filter_len, activation='relu')(inputs)
	X = Dropout(dr1)(X)
	X = GRU(rnn_len,return_sequences=True)(X)
	X = Dropout(dr2)(X)

	X = Flatten()(X)

	X = Dense(FFL)(X)
	X = Activation('relu')(X)
	X = Dropout(dr3)(X)
	Y = Dense(1,activation='sigmoid', kernel_initializer='uniform')(X)

	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	his = model.fit(X_train, y_train, epochs=EPOCH, batch_size=256, shuffle=True,
							validation_data=(X_test, y_test) )
	y_pred = model.predict(X_test)
	rslt = print_result(y_test, y_pred, desc)
	return rslt		# for CV test 
# end of smi_model_train
	

def smi_model_train_test( X_train, y_train, X_test, y_test, desc, params ) :
	EPOCH, FFL, n_filters, filter_len, rnn_len, dr1, dr2, dr3, indim= params
	dr1=0.25
	params = [EPOCH, FFL, n_filters, filter_len, rnn_len, dr1, dr2, dr3]
	desc += str(params)

	print('SMI Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	inputs = Input(shape=(X_train.shape[1:]))

	X = LSTM(10, use_bias=False)(inputs)

	X = Dense(FFL)(X)
	X = Activation('relu')(X)
	X = Dropout(dr3)(X)
	Y = Dense(1,activation='sigmoid', kernel_initializer='uniform')(X)


	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	his = model.fit(X_train, y_train, epochs=100, batch_size=256, shuffle=True,
							validation_data=(X_test, y_test) )
	y_pred = model.predict(X_test)
	rslt = print_result(y_test, y_pred, desc)
	return rslt		# for CV test 
# end of smi_model_train


def smi_reslstm( X_train, y_train, X_test, y_test, desc, params ) :
	EPOCH, FFL, n_filters, filter_len, rnn_len, dr1, dr2, dr3, indim= params
	desc += str(params)

	print('SMI Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	inputs = Input(shape=(X_train.shape[1:]))
	masked = Masking(0.)(inputs)

	x = masked
	rnn_width=21
	rnn_dropout=0.1
	rnn_depth=10
	for i in range(rnn_depth):
		x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True)(x)
		x = add([x, x_rnn])
	X=x
	X = LSTM(rnn_len)(X)
	Y = Dense(1,activation='sigmoid', kernel_initializer='uniform')(X)

	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	his = model.fit(X_train, y_train, epochs=300, batch_size=256, shuffle=True,
							validation_data=(X_test, y_test) )
	y_pred = model.predict(X_test)
	rslt = print_result(y_test, y_pred, desc)
	return rslt		# for CV test 
# end of smi_model_train


def smi_model_train_bagging( X_train, y_train, X_test, y_test, desc, params, MAXITR ) :
	EPOCH, FFL, n_filters, filter_len, rnn_len, dr1, dr2, dr3, indim= params
	desc += str(params)

	print('SMI Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	inputs = Input(shape=(X_train.shape[1:]))

	X = Convolution1D(n_filters,filter_len, activation='relu')(inputs)
	X = Dropout(dr1)(X)
	X = GRU(rnn_len,return_sequences=True)(X)
	X = Dropout(dr2)(X)

	X = Flatten()(X)

	X = Dense(FFL)(X)
	X = Activation('relu')(X)
	X = Dropout(dr3)(X)
	Y = Dense(1,activation='sigmoid', kernel_initializer='uniform')(X)

	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	his = model.fit(X_train, y_train, epochs=EPOCH, batch_size=256, shuffle=True,
							validation_data=(X_test, y_test) )
	y_pred = model.predict(X_test)
	rslt = print_result(y_test, y_pred, desc)

	y_pred = np.zeros( (MAXITR, len(X_test), 1) )
	Wsave = model.get_weights()
	for i in range(MAXITR) :
		print "bagging idx:",i
		model.set_weights(Wsave)
		model.reset_states()
		model = Varweightfit(X_train, y_train, X_test, y_test, desc, EPOCH, model)
		y_pred[i]=model.predict(X_test)
	y_mean = np.mean(y_pred, axis=0)
	rslt = print_result(y_test, y_mean, desc)

	return rslt		# for CV test 
# end of smi_model_train
	

#---------------------------------------------------------------------------
# 881 Fingerprint Model Building
#---------------------------------------------------------------------------
def fp_model_train_bagging( X_train, y_train, X_test, y_test, MAXITR, desc ) :
	print('FP Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	EPOCH=30

	inputs = Input(shape=(X_train.shape[1:]))

	X = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(inputs)
	X = Dropout(0.6)(X)
	X = Dense(64)(X)
	X = Dropout(0.1)(X)
	X = Activation('tanh')(X)

	Y = Dense(1, activation='sigmoid', kernel_initializer='uniform')(X)

	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	y_pred = np.zeros( (MAXITR, len(X_test), 1) )
	Wsave = model.get_weights()
	for i in range(MAXITR) :
		print "bagging idx:",i
		model.set_weights(Wsave)
		model.reset_states()
		model = Varweightfit(X_train, y_train, X_test, y_test, desc, EPOCH, model)
		y_pred[i]=model.predict(X_test)
	y_mean = np.mean(y_pred, axis=0)
	rslt = print_result(y_test, y_mean, desc)
	return rslt 
# end of fp_model_train


def fp_model_train_repeat( X_train, y_train, X_test, y_test, MAXITR, desc ) :
	print('FP Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	EPOCH=30

	inputs = Input(shape=(X_train.shape[1:]))

	X = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(inputs)
	X = Dropout(0.6)(X)
	X = Dense(64)(X)
	X = Dropout(0.1)(X)
	X = Activation('tanh')(X)

	Y = Dense(1, activation='sigmoid', kernel_initializer='uniform')(X)

	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	y_pred = np.zeros( (MAXITR, len(X_test), 1) )
	Wsave = model.get_weights()
	for i in range(MAXITR) :
		print "repeat idx:",i
		model.set_weights(Wsave)
		model.reset_states()
		his = model.fit(X_train, y_train, epochs=EPOCH, batch_size=256, shuffle=True,
								validation_data=(X_test, y_test) )
		y_pred[i] = model.predict(X_test)

	y_mean = np.mean(y_pred, axis=0)
	rslt = print_result(y_test, y_mean, desc)
	return rslt 
# end of fp_model_train


def fp_model_train( X_train, y_train, X_test, y_test, desc ) :
	print('FP Build model...')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	EPOCH=30

	inputs = Input(shape=(X_train.shape[1:]))

	X = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(inputs)
	X = Dropout(0.6)(X)
	X = Dense(64)(X)
	X = Dropout(0.1)(X)
	X = Activation('tanh')(X)

	Y = Dense(1, activation='sigmoid', kernel_initializer='uniform')(X)

	model = Model(inputs=inputs, outputs=Y)
	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	his = model.fit(X_train, y_train, epochs=EPOCH, batch_size=256, shuffle=True,
							validation_data=(X_test, y_test) )
	y_pred = model.predict(X_test)
	rslt = print_result(y_test, y_pred, desc)
	return rslt 
# end of fp_model_train

