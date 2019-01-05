from admm_nmf import ADMM_NMF
from parser import parameter_parser
from utils import read_features, tab_printer

def execute_factorization():
    """
    Reading the target matrix, running optimization and saving to hard drive.
    """
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
