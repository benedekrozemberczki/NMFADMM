from utils import read_features
from admm_nmf import ADMM_NMF
from parser import parameter_parser

def execute_factorization():
    args = parameter_parser()
    X = read_features(args.input_path)
    model = ADMM_NMF(X, args)
    model.optimize()
    model.save_user_factors()
    model.save_item_factors()

if __name__ == '__main__':

    execute_factorization()

