import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives a factorization of the Twitch/Taiwan dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by user/item ID.
    """

    parser = argparse.ArgumentParser(description = "Run ADMM-NMF.")

    parser.add_argument("--input-path",
                        nargs = "?",
                        default = "./input/twitch_taiwan.csv",
	                help = "Input csv.")

    parser.add_argument("--user-path",
                        nargs = "?",
                        default = "./output/twitch_taiwan_user.csv",
	                help = "User factors csv.")

    parser.add_argument("--item-path",
                        nargs = "?",
                        default = "./output/twitch_taiwan_item.csv",
	                help = "Item factors csv.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
	                help = "Number of training epochs. Default is 100.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 32,
	                help = "Number of dimensions. Default is 32.")

    parser.add_argument("--rho",
                        type = float,
                        default = 1.0,
	                help = "Regularization parameter. Default is 1.0.")

    return parser.parse_args()
