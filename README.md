# FROCC (Fast Random projections based One Class Classifier)

## Running PardFROCC

`PardFROCC` (and other versions) adheres to `sklearn` estimator interface. `PardFROCC` and `Sparse DFROCC` expects sparse matrices in CSR format. Other methods expect `numpy` arrays.

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
## Running synthetic generated data

 ```
 python experiment.py --dataset <DATASET> --epsilon <EPSILON> --dimension <DIMENSION> --n_samples <N_SAMPLES> --n_dim <N_DIM> --method <METHOD> --repetitions <REPETITIONS> --outfile <OUTFILE>
 ```

 where

 `dataset` - one amongst ``mmgauss`` and ``himoon``

 `epsilon` - Seperation parameter. Typically of the order 0.1-0.0001 (Default: 0.01)

 `dimension` - Number of FROCC dimensions (Default: 1000)

 `n_samples` - Number of samples to generate *when using generated data* (Default: 1000)

 `n_dim` - Number of dimensions in the data to be generated *when using generated data* (Default: 1000)

 `method` - One amongst `frocc`, `dfrocc`, `sparse_dfrocc`, `pardfrocc` (Default: `pardfrocc`)

 `repetations` - Number of repetitions to run the experiment for (Default: 1)

 `outfile` - File path to write results to (Default: `results/out.csv`)

 Example:
 ```
 python experiment.py --dataset himoon --epsilon 0.01 --dimension 100 --n_samples 100000 --n_dim 100000 --method pardfrocc --repetitions 1 --outfile himoon_results.csv
 ```
