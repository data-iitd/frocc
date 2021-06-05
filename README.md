# FROCC (Fast Random projections based One Class Classifier)

## Running PardFROCC

`PardFROCC` (and other versions) adheres to `sklearn` estimator interface. `PardFROCC` and `Sparse DFROCC` expects sparse matrices in CSC format. Other methods expect `numpy` arrays. The arrays can be `float` or `int` types.

```python
import pardfrocc
from sklearn.metrics import roc_auc_score

x_train, x_test, y_test = load_data() # any function to load data in appropriate format
clf = pardfrocc.ParDFROCC()
clf.fit(x_train)

scores = clf.decision_function(x_test)
roc = roc_auc_score(y_test, scores)

predictions = clf.predict(x_test)
```

Any numeric data file can be converted to required format as follows:

```python
import numpy as np
import scipy.sparse as sp

x = np.array(x, dtype=np.float32) # for dense data
x = sp.csc_matric(x, dtype=np.float32) # for sparse data
```

For more information of data loading, see [[Sparse matrix docs](https://docs.scipy.org/doc/scipy/reference/sparse.html)] and [[Numpy array docs](https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.html)].

## Generating synthetic data

We provide a script for generating HiMoon and MMGauss dataset.

```python
import data_gen
x_train, y_train, x_val, y_val, x_test, y_test = data_gen.himoon(n_samples=1000, n_dims=1000) #or data_gen.mmgauss()
```

Parameters:

`n_samples` - Total number of generated samples (Default: 1000)

`n_dims` - Number of dimensions of generated data  (Default: 1000)

`sparsity` - Sparsity of the generated data  (Default: 0.01)

## Running synthetic generated data

 ```
 python experiment.py --dataset <DATASET> --epsilon <EPSILON> --dimension <DIMENSION> --n_samples <N_SAMPLES> --n_dim <N_DIM> --method <METHOD> --repetitions <REPETITIONS> --outfile <OUTFILE>
 ```

 where

 `dataset` - one amongst ``mmgauss`` and ``himoon``

 `epsilon` - Seperation parameter. Typically of the order 0.1-0.0001 (Default: 0.01)

 `dimension` - Number of FROCC dimensions (Default: 1000)

`kernel`

 `n_samples` - Number of samples to generate *when using generated data* (Default: 1000)

 `n_dim` - Number of dimensions in the data to be generated *when using generated data* (Default: 1000)

 `method` - One amongst `frocc`, `dfrocc`, `sparse_dfrocc`, `pardfrocc` (Default: `pardfrocc`)

 `repetations` - Number of repetitions to run the experiment for (Default: 1)

 `outfile` - File path to write results to (Default: `results/out.csv`)

 Example:
 ```
 python experiment.py --dataset himoon --epsilon 0.01 --dimension 100 --n_samples 100000 --n_dim 100000 --method pardfrocc --repetitions 1 --outfile himoon_results.csv
 ```
## Hyper-parameter selection

Details of hyper-parameters used to reproduce the results is provided in [hyper.pdf](./hyper.pdf)

## Disclaimer
This is released as a research prototype. It is not meant to be a production quality implementation. It has been made open source to enable easy reproducibility of research results.
