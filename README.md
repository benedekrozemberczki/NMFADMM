Non-negative Matrix Factorization with Alternating Direction Method of Multipliers
============================================
A sparsity aware implementation of "Alternating Direction Method of Multipliers for Non-Negative Matrix Factorization with the Beta-Divergence"  (ICASSP 2014).

<div style="text-align:center"><img src ="sgcn.jpg" ,width=600/></div>
<p align="justify">
Non-negative matrix factorization (NMF) is a popular method for learning interpretable features from non-negative data, such as counts or magnitudes. Different cost functions are used with NMF in different applications. We develop an algorithm, based on the alternating direction method of multipliers, that tackles NMF problems whose cost function is a beta-divergence, a broad class of divergence functions. We derive simple, closed-form updates for the most commonly used beta-divergences. We demonstrate experimentally that this algorithm has faster convergence and yields superior results to state-of-the-art algorithms for this problem.</p>

This repository provides an implementation for ADMM NMF as described in the paper:

> Alternating Direction Method of Multipliers for Non-Negative Matrix Factorization with the Beta-Divergence.
> Dennis L. Sun and Cédric Févotte 
> ICASSP 2014
> [[Paper]](http://statweb.stanford.edu/~dlsun/papers/nmf_admm.pdf)


### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
tqdm               4.28.1
numpy              1.15.4
pandas             0.23.4
texttable          1.5.0
scipy              1.1.0
argparse           1.1.0
```
### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Bitcoin Alpha`  and `Bitcoin OTC` graphs are included in the  `input/` directory. The structure of the edge dataset is the following:

| **NODE ID 1**| **NODE ID 2** | **Sign** | 
| --- | --- | --- |
| 0 | 3 |-1 |
| 1 | 1 |1 |
| 2 | 2 |1 |
| 3 | 1 |-1 |
| ... | ... |... |
| n | 9 |-1 |

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path                STR    Input graph path.          Default is `input/bitcoin_otc.csv`.
  --features-path            STR    Membership path.           Default is `input/bitcoin_otc.csv`.
  --embedding-path           STR    Embedding path.            Default is `output/embedding/bitcoin_otc_sgcn.csv`.
  --regression-weights-path  STR    Regression weights path.   Default is `output/weights/bitcoin_otc_sgcn.csv`.
  --log-path                 STR    Log path.                  Default is `logs/bitcoin_otc_logs.json`.  
```

#### Model options

```
  --epochs                INT         Number of SGCN training epochs.      Default is 100. 
  --reduction-iterations  INT         Number of SVD epochs.                Default is 128.
  --reduction-dimensions  INT         SVD dimensions.                      Default is 30.
  --seed                  INT         Random seed value.                   Default is 42.
  --lamb                  FLOAT       Embedding regularization parameter.  Default is 1.0.
  --test-size             FLOAT       Test ratio..                         Default is False.  
  --learning-rate         FLOAT       Learning rate.                       Default is 0.001.  
  --weight-decay          FLOAT       Weight decay.                        Default is 10^-5. 
  --layers                LST         Layer sizes in model.                Default is [64, 32].
  --spectral-features     BOOL        Layer sizes in autoencoder model.    Default is True
  --general-features      BOOL        Loss calculation for the model.      Sets spectral features to False.  
```

### Examples

The following commands learn a node embedding, regression weights and write the embedding to disk. The node representations are ordered by the ID. The layer sizes can be set manually.

Training an SGCN model on the default dataset. Saving the embedding, regression weights and logs at default paths.
```
python src/main.py
```
<p align="center">
<img style="float: center;" src="sgcn_run_example.jpg">
</p>

Creating an SGCN model of the default dataset with a 96-64-32 architecture.
```
python src/main.py --layers 96 64 32
```
Creating a single layer SGCN model with 32 features.
```
python src/main.py --layers 32
```
Creating a model with some custom learning rate and epoch number.
```
python src/main.py --learning-rate 0.001 --epochs 200
```
Training a model of another dataset with features present - a signed `Erdos-Renyi` graph. Saving the weight output and logs in a custom folder.
```
python src/main.py --general-features --edge-path input/erdos_renyi_edges.csv --features-path input/erdos_renyi_features.csv --embedding-path output/embedding/erdos_renyi.csv --regression-weights-path output/weights/erdos_renyi.csv --log-path logs/erdos_renyi.json
```
