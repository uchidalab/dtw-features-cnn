# ---------GENERIC--------------

C1_LAYER_SIZE = 64
C2_LAYER_SIZE = 128
C3_LAYER_SIZE = 256
C4_LAYER_SIZE = 512
FC_LAYER_SIZE = 1024
 
NUM_ITER = 100000
BATCH_SIZE = 100
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0001


def load_settings_raw(dataset, dimen):
	global TRAINING_FILE
	global TEST_FILE
	global TRAINING_LABEL
	global TEST_LABEL
	global NUM_CLASSES
	global IMAGE_SHAPE
	global CONV_OUTPUT_SHAPE
	global MPOOL_SHAPE
	
	TRAINING_FILE = "data/raw-train-data-{0}.txt".format(dataset)
	TEST_FILE = "data/raw-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL = "data/test-label-{0}.txt".format(dataset)

	if dataset == '1a':
		NUM_CLASSES = 10
	else:
		NUM_CLASSES = 26
	
	if dimen == '1d':
		CONV_OUTPUT_SHAPE = 7 #50 25 13 7
		MPOOL_SHAPE = 2
		IMAGE_SHAPE = (50, 2) 
	else:
		CONV_OUTPUT_SHAPE = 7*2 #50 25 13 7
		MPOOL_SHAPE = (2,1)
		IMAGE_SHAPE = (50, 2, 1)

def load_settings_dtwfeatures(dataset, dimen):
	global TRAINING_FILE
	global TEST_FILE
	global TRAINING_LABEL
	global TEST_LABEL
	global NUM_CLASSES
	global IMAGE_SHAPE
	global CONV_OUTPUT_SHAPE
	global MPOOL_SHAPE

	TRAINING_FILE = "data/dtw_features-50-train-data-{0}.txt".format(dataset)
	TEST_FILE = "data/dtw_features-50-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL = "data/test-label-{0}.txt".format(dataset)

	if dataset == '1a':
		NUM_CLASSES = 10
	else:
		NUM_CLASSES = 26
	
	if dimen == '1d':
		CONV_OUTPUT_SHAPE = 7 #50 25 13 7
		MPOOL_SHAPE = 2
		if dataset == '1a':
			IMAGE_SHAPE = (50, 50) 
		else:
			IMAGE_SHAPE = (50, 52) 
	else:
		MPOOL_SHAPE = (2,1)
		if dataset == '1a':
			IMAGE_SHAPE = (50, 50, 1) 
			CONV_OUTPUT_SHAPE = 7*50 #50 25 13 7
		else:
			IMAGE_SHAPE = (50, 52, 1) 
			CONV_OUTPUT_SHAPE = 7*52 #50 25 13 7

def load_settings_early(dataset, dimen):
	global TRAINING_FILE
	global TEST_FILE
	global TRAINING_LABEL
	global TEST_LABEL
	global NUM_CLASSES
	global IMAGE_SHAPE
	global CONV_OUTPUT_SHAPE
	global MPOOL_SHAPE

	TRAINING_FILE = "data/dtw_features-50-plus-raw-train-data-{0}.txt".format(dataset)
	TEST_FILE = "data/dtw_features-50-plus-raw-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL = "data/test-label-{0}.txt".format(dataset)

	if dataset == '1a':
		NUM_CLASSES = 10
	else:
		NUM_CLASSES = 26

	if dimen == '1d':
		CONV_OUTPUT_SHAPE = 7 #50 25 13 7
		MPOOL_SHAPE = 2
		if dataset == '1a':
			IMAGE_SHAPE = (50, 52) 
		else:
			IMAGE_SHAPE = (50, 54) 
	else:
		MPOOL_SHAPE = (2,1)
		if dataset == '1a':
			IMAGE_SHAPE = (50, 52, 1) 
			CONV_OUTPUT_SHAPE = 7*52 #50 25 13 7
		else:
			IMAGE_SHAPE = (50, 54, 1) 
			CONV_OUTPUT_SHAPE = 7*54 #50 25 13 7



def load_settings_mid(dataset, dimen):
	global TRAINING_FILE1
	global TEST_FILE1
	global TRAINING_LABEL1
	global TEST_LABEL1
	global IMAGE_SHAPE1

	global TRAINING_FILE2
	global TEST_FILE2
	global TRAINING_LABEL2
	global TEST_LABEL2
	global IMAGE_SHAPE2

	global CONV_OUTPUT_SHAPE

	global NUM_CLASSES
	global MPOOL_SHAPE

	TRAINING_FILE1 = "data/raw-train-data-{0}.txt".format(dataset)
	TEST_FILE1 = "data/raw-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL1 = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL1 = "data/test-label-{0}.txt".format(dataset)

	TRAINING_FILE2 = "data/dtw_features-50-train-data-{0}.txt".format(dataset)
	TEST_FILE2 = "data/dtw_features-50-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL2 = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL2 = "data/test-label-{0}.txt".format(dataset)

	if dataset == '1a':
		NUM_CLASSES = 10
	else:
		NUM_CLASSES = 26

	if dimen == '1d':
		CONV_OUTPUT_SHAPE = 7*2 #50 25 13 7
		MPOOL_SHAPE = 2
		if dataset == '1a':
			IMAGE_SHAPE1 = (50, 2) 
			IMAGE_SHAPE2 = (50, 50)
		else:
			IMAGE_SHAPE1 = (50, 2) 
			IMAGE_SHAPE2 = (50, 52)
	else:
		MPOOL_SHAPE = (2,1)
		if dataset == '1a':
			IMAGE_SHAPE1 = (50, 2, 1) 
			IMAGE_SHAPE2 = (50, 50, 1) 
			CONV_OUTPUT_SHAPE = (7*2)+(7*50)
		else:
			IMAGE_SHAPE1 = (50, 2, 1) 
			IMAGE_SHAPE2 = (50, 52, 1)
			CONV_OUTPUT_SHAPE = (7*2)+(7*52)



def load_settings_late(dataset, dimen):
	global TRAINING_FILE1
	global TEST_FILE1
	global TRAINING_LABEL1
	global TEST_LABEL1
	global IMAGE_SHAPE1
	global CONV_OUTPUT_SHAPE1

	global TRAINING_FILE2
	global TEST_FILE2
	global TRAINING_LABEL2
	global TEST_LABEL2
	global IMAGE_SHAPE2
	global CONV_OUTPUT_SHAPE2

	global NUM_CLASSES
	global MPOOL_SHAPE

	TRAINING_FILE1 = "data/raw-train-data-{0}.txt".format(dataset)
	TEST_FILE1 = "data/raw-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL1 = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL1 = "data/test-label-{0}.txt".format(dataset)

	TRAINING_FILE2 = "data/dtw_features-50-train-data-{0}.txt".format(dataset)
	TEST_FILE2 = "data/dtw_features-50-test-data-{0}.txt".format(dataset)
	TRAINING_LABEL2 = "data/train-label-{0}.txt".format(dataset)
	TEST_LABEL2 = "data/test-label-{0}.txt".format(dataset)

	if dataset == '1a':
		NUM_CLASSES = 10
	else:
		NUM_CLASSES = 26

	if dimen == '1d':
		CONV_OUTPUT_SHAPE1 = 7 #50 25 13 7
		CONV_OUTPUT_SHAPE2 = 7 #50 25 13 7
		MPOOL_SHAPE = 2
		if dataset == '1a':
			IMAGE_SHAPE1 = (50, 2) 
			IMAGE_SHAPE2 = (50, 50)
		else:
			IMAGE_SHAPE1 = (50, 2) 
			IMAGE_SHAPE2 = (50, 52)
	else:
		MPOOL_SHAPE = (2,1)
		if dataset == '1a':
			IMAGE_SHAPE1 = (50, 2, 1) 
			IMAGE_SHAPE2 = (50, 50, 1) 
			CONV_OUTPUT_SHAPE1 = 7*2 #50 25 13 7
			CONV_OUTPUT_SHAPE2 = 7*50 #50 25 13 7
		else:
			IMAGE_SHAPE1 = (50, 2, 1) 
			IMAGE_SHAPE2 = (50, 52, 1)
			CONV_OUTPUT_SHAPE1 = 7*2 #50 25 13 7
			CONV_OUTPUT_SHAPE2 = 7*52 #50 25 13 7


