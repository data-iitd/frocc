import argparse
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import data_gen
import frocc
import dfrocc
import sparse_dfrocc
import pardfrocc
import kernels as k

# import utils

parser = argparse.ArgumentParser()

parser.add_argument("--dataset")
parser.add_argument("--dimension", default=1000, type=int)
parser.add_argument("--epsilon", default=0.01, type=np.float)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--repetitions", default=1, type=int)
parser.add_argument("--outfile", default="results/out.csv")
parser.add_argument("--method", default="pardfrocc")
parser.add_argument("--n_samples", default=1000, type=int)
parser.add_argument("--n_dims", default=1000, type=int)

args = parser.parse_args()

if args.dataset == "himoon":
    x, y, _, _, xtest, ytest = data_gen.himoon(
        n_samples=args.n_samples, n_dims=args.n_dims
    )

elif args.dataset == "mmgauss":
    x, y, _, _, xtest, ytest = data_gen.mmgauss(
        n_samples=args.n_samples, n_dims=args.n_dims
    )
else:
    raise ValueError("Unknown dataset")

kernels = dict(
    zip(
        ["rbf", "linear", "poly", "sigmoid"],
        [k.rbf(), k.linear(), k.poly(), k.sigmoid()],
    )
)
try:
    kernel = kernels.get(args.kernel)
except KeyError as e:
    kernel = "linear"

df = pd.DataFrame()

print(
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Running "
    + f"{args.dataset} with {args.dimension} dimensions and "
    + f"epsilon={args.epsilon} with {args.kernel} kernel for "
    + f"{args.repetitions} repetitions."
)

for run in range(args.repetitions):
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run {run + 1} of "
        + f"{args.repetitions}"
    )
    if args.method == "frocc":
        clf = frocc.FROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )
        x = x.toarray()
        xtest = xtest.toarray()
    elif args.method == "dfrocc":
        clf = dfrocc.DFROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )
    elif args.method == "sparse_dfrocc":
        clf = sparse_dfrocc.SDFROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )
    elif args.method == "pardfrocc":
        clf = pardfrocc.ParDFROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )

    tic = time()
    clf.fit(x)
    train_time = (time() - tic) * 1000 / x.shape[0]
    tic = time()
    scores = clf.decision_function(xtest)
    test_time = (time() - tic) * 1000 / xtest.shape[0]
    roc = roc_auc_score(ytest, scores)
    df = df.append(
        {
            "Run ID": run,
            "Dimension": args.dimension,
            "Epsilon": args.epsilon,
            "AUC of ROC": roc,
            "Train Time": train_time,
            "Test Time": test_time,
        },
        ignore_index=True,
    )
df.to_csv(args.outfile)
