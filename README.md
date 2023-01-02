# Installation
```
pip install torch transformers tensorboard pandas scikit-learn tqdm
```

# Notes
During the first run, feature caching and evaluation requires a lot of CPU memory (>=150 GB)
and will
save about 25 GB of files on the hard disk.
This CPU requirement can be reduced, at the expense of speed,
by lowering the `--threads` count in `run_maud.sh`.

Training uses around 22 GB of GPU memory.


# Validation runs

`./run_maud.sh`

# Test runs (using best-found hps)

`./run_maud_best_hp.sh`
