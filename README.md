# ReSeq

This is the official PyTorch implementation for the paper:

> Reciprocal Sequential Recommendation

## Overview

We proposes a reciprocal sequential recommendation method, named ReSeq, in which we formulate reciprocal recommendation as a distinctive sequence matching task and perform matching prediction based on bilateral dynamic behavior sequences.![model](./asset/model.jpg)

## Requirements

```
torch==1.10.1+cu113
cudatoolkit==11.3
```

### Run

```
cd ./run
python auto_run.py
```

