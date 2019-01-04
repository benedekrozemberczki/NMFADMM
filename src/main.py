from utils import read_features, tab_printer
from admm_nmf import ADMM_NMF
from parser import parameter_parser

def execute_factorization():
    args = parameter_parser()
    tab_printer(args)
    X = read_features(args.input_path)
    print("\nTraining started.\n")
    model = ADMM_NMF(X, args)
    model.optimize()
    print("\nFactors saved.\n")
    model.save_user_factors()
    model.save_item_factors()

if __name__ == '__main__':

    execute_factorization()

