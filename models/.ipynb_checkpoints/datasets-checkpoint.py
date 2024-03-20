import h5py
import pickle
import numpy as np
import torch

def loadRadioML2016(dataDir): 
    with open(dataDir, "rb") as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
      X = []
      Y = []
      Z = []
      modClasses = set()
      SNRs = set()
      for modClass,SNR in data.keys():
          modClasses.add(modClass)
          SNRs.add(SNR)
      modClasses = list(modClasses)
      SNRs = list(SNRs)
      SNRs.sort()
      for modClass, SNR in data.keys():
          x = np.array(data[(modClass,SNR)])
          label = modClasses.index(modClass) # Get the Index Label
          label_h = np.array([1 if i==label else 0 for i in range(len(modClasses))]) # One-Hot Encode
          y = np.tile(label_h,(x.shape[0],1))
          z = np.tile(SNR,(x.shape[0],1))
          X.append(x)
          Y.append(y)
          Z.append(z)
      X = np.vstack(X)
      X = X.transpose(0,2,1)
      Y = np.vstack(Y)
      Z = np.vstack(Z)
    return X,Y,Z,modClasses,SNRs

# Load RadioML 2018.01 Dataset
def loadRadioML2018(dataDir): 
    with h5py.File(dataDir, "r") as f:
        # X is IQ Signals
        X = f.get('X')[:]
        # Y is Modulation Classes
        Y = f.get('Y')[:]
        # Z is SNR Ratios
        Z = f.get('Z')[:]
        modClasses = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK',
        '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK',
        '128APSK', '16QAM', '32QAM', '64QAM', '128QAM',
        '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
        'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        SNRs = np.unique(Z)
        return X,Y,Z,modClasses,SNRs

# Load Hisar Dataset
def loadHisar():
    pass

# Prepare Datset for Training
def prepareDataset(X,Y,Z,split=[0.9,0.05,0.05],batch_size=9000):
    N = X.shape[0]
    indexes = np.arange(N)
    np.random.seed(0) # Make sure it's repeatable
    np.random.shuffle(indexes)
    train_indexes = indexes[0:int(split[0]*N)]
    val_indexes = indexes[int(split[0]*N):int((split[0]+split[1])*N)]
    test_indexes = indexes[int((split[0]+split[1])*N):N]
    
    X = X.transpose(0,2,1)
    # Loading Specific Subsets
    X_train = torch.tensor(X[train_indexes],dtype=torch.float32)
    X_val = torch.tensor(X[val_indexes],dtype=torch.float32)
    X_test = torch.tensor(X[test_indexes],dtype=torch.float32)
    
    Y_train = torch.tensor(Y[train_indexes],dtype=torch.float32)
    Y_val = torch.tensor(Y[val_indexes],dtype=torch.float32)
    Y_test = torch.tensor(Y[test_indexes],dtype=torch.float32)

    Z_test = Z[test_indexes]
    
    train_data = torch.utils.data.TensorDataset(X_train,Y_train)
    val_data = torch.utils.data.TensorDataset(X_val,Y_val)
    test_data = torch.utils.data.TensorDataset(X_test,Y_test)
    
    # Creating Data Loaders
    train_loader = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data,shuffle=False,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=batch_size)

    return train_data,val_data,test_data,train_loader,val_loader,test_loader,Z_test


# Clears all the Dataset Variables in Memory 
def clearDatasetVars(tensors=False):
    datasetVars = ['X','Y','Z','X_train','Y_train','X_val','Y_val','X_test','Y_test']
    if(tensors): # Add Tensor Variables Needed for Model Training
        tensorVars = ['train_data','valid_data','test_data','train_loader','val_loader','test_loader','model','best_model','Z_test']
        datasetVars += tensorVars
    for datasetVar in datasetVars: # Delete in Global Variables if it Exists
        if datasetVar in globals():
            del globals()[datasetVar]
    if(torch.cuda.is_available()): # Free GPU Memory
        torch.cuda.empty_cache()
    
