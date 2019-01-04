import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

def read_features(path):

    features = pd.read_csv(path)

    index_1 = features["index_1"].values.tolist()
    index_2 = features["index_2"].values.tolist()
    values = features["values"].values.tolist()

    users = max(index_1)+1
    items = max(index_2)+1

    features = sparse.csr_matrix(sparse.coo_matrix((values,(index_1,index_2)),shape=(users,items),dtype=np.float32))
    return features
