import json
import numpy as np
import pandas as pd
from scipy import sparse
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def read_features(path):
    """
    Reading the sparse matrix.
    :param path: Path to the sparse matrix.
    :Return features: Target matrix.
    """
    features = pd.read_csv(path)
    
    index_1 = features["index_1"].values.tolist()
    index_2 = features["index_2"].values.tolist()
    values = features["values"].values.tolist()
    
    users = max(index_1)+1
    items = max(index_2)+1
    
    features = sparse.csr_matrix(sparse.coo_matrix((values,(index_1,index_2)),shape=(users,items),dtype=np.float32))
    return features
