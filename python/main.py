import gzip
import numpy as np
from random_forest import RandomForest


if __name__=='__main__':
    # Load data
    with gzip.open('../data/train-labels-idx1-ubyte.gz', 'rb') as lbpath:
        label_train = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                    offset=8)
    with gzip.open('../data/train-images-idx3-ubyte.gz', 'rb') as imgpath:
        data_train = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                    offset=16).reshape(len(label_train), 784)
    with gzip.open('../data/t10k-labels-idx1-ubyte.gz', 'rb') as lbpath:
        label_test = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8) 
    with gzip.open('../data/t10k-images-idx3-ubyte.gz', 'rb') as imgpath:
        data_test = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(label_test), 784)

    rf = RandomForest()
    rf.fit(data_train, label_train)
    predict = rf.predict(data_test)
    
    print('Accuracy: ', sum(predict == label_test)/label_test.shape[0])
