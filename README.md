# MAUD Supplementary Extraction Task

This repository contains the dataset and baseline code for the MAUD supplementary
extraction task, as described in the appendix of
"MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding".

For the main MAUD dataset and baselines, see [github.com/TheAtticusProject/maud](http://github.com/TheAtticusProject/maud).

## Installation
```
pip install torch transformers tensorboard pandas scikit-learn tqdm
```

## Notes
During the first run, feature caching and evaluation requires a lot of CPU memory (>=150 GB)
and will
save about 25 GB of files on the hard disk.
This CPU requirement can be reduced, at the expense of speed,
by lowering the `--threads` count in `run_maud.sh`.

Training uses around 22 GB of GPU memory.


## Validation runs (grid-search)

`./run_maud.sh`

## Test runs with best-performing hyperparameters

`./run_maud_best_hp.sh`
