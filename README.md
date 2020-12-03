# FROCC (Fast Random projections based One Class Classifier)
 
 ```
 python experiment.py --dataset <DATASET> --epsilon <EPSILON> --dimension <DIMENSION> --n_samples <N_SAMPLES> --n_dim <N_DIM> --method <METHOD> --repetitions <REPETITIONS> --outfile <OUTFILE>
 ```
 
 where 
 
 <DATASET> - one amongst ``mmgauss`` and ``himoon``
 
 <EPSILON> - Seperation parameter. Typically of the order 0.1-0.0001 (Default: 0.01)
 
 <DIMENSION> - Number of FROCC dimensions (Default: 1000)
 
 <N_SAMPLES> - Number of samples to generate (Default: 1000)
 
 <N_DIM> - Number of dimensions in the data to be generated (Default: 1000)
 
 <METHOD> - One amongst `frocc`, `dfrocc`, `sparse_dfrocc`, `pardfrocc` (Default: `pardfrocc`)
 
 <REPETITIONS> - Number of repetitions to run the experiment for (Default: 1)
 
 <OUTFILE> - File path to write results to (Default: `results/out.csv`)
 
 Example:
 ```
 python experiment.py --dataset himoon --epsilon 0.01 --dimension 100 --n_samples 100000 --n_dim 100000 --method pardfrocc --repetitions 1 --outfile himoon_results.csv
 ```
