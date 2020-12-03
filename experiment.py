import argparse
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import datasets
import frocc_bool
import kernels as k
# import utils

parser = argparse.ArgumentParser()

parser.add_argument("--dataset")
parser.add_argument("--dimension", default=1000, type=int)
parser.add_argument("--epsilon", default=0.01, type=np.float)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--repetitions", default=3, type=int)
parser.add_argument("--outfile", default="results/out.csv")

args = parser.parse_args()

if args.dataset == "mnist":
    x, y, xtest, ytest = datasets.mnist()
elif args.dataset == "cifar":
    x, y, xtest, ytest = datasets.cifar()
elif args.dataset == "cifar_100":
    x, y, xtest, ytest = datasets.cifar_100()
elif args.dataset == "omniglot":
    x, y, xtest, ytest = datasets.omniglot()
elif args.dataset == "miniboone":
    x, y, xtest, ytest = datasets.miniboone()
elif args.dataset == "magic_telescope":
    x, y, xtest, ytest = datasets.magic_telescope()
elif args.dataset == "diabetes":
    x, y, xtest, ytest = datasets.diabetes()
elif args.dataset == "vehicle":
    x, y, xtest, ytest = datasets.vehicle()
elif args.dataset == "cardio":
    x, y, xtest, ytest = datasets.cardio()
else:
    raise ValueError("Unknows dataset")

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
    print(e)

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
    clf = frocc_bool.FROCC(
        num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
    )
    tic = time()
    clf.fit(x)
    train_time = (time() - tic) * 1000 / len(x)
    tic = time()
    scores = clf.decision_function(xtest)
    test_time = (time() - tic) * 1000 / len(xtest)
    roc = roc_auc_score(ytest, scores)
    # patn = utils.precision_at_n_score(ytest, scores)
    df = df.append(
        {
            "Run ID": run,
            "Dimension": args.dimension,
            "Epsilon": args.epsilon,
            "ROC": roc,
            # "P@n": patn,
            "Train Time": train_time,
            "Test Time": test_time,
        },
        ignore_index=True,
    )
df.to_csv(args.outfile)
