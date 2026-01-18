# Controlling Neural Network Generalization via Constraint-Guided Weight Transformations
--

## Requirements

- Python (≥3.7)
- PyTorch
- Gurobi Optimizer (we used the academic license)
  - Install Gurobi following instructions at: https://www.gurobi.com/documentation/

##  Training Accuracy-preserving Generalization Degradation (TAGD)

You can run TAGD on either image or tabular datasets.

### For Image Datasets:

```
cd Image
python Main.py TAGD
```

### For Tabular Datasets:

```
cd Tabular
python Main.py TAGD
```

##  Controlled Misclassification (CMC)

You can run CMC on either image or tabular datasets.

### For Image Datasets:

```
cd Image
python Main.py CMC <Save Checkpoint(Y/N)> <Misclassification Count> <Misclassification Type>
```

### For Tabular Datasets:

```
cd Tabular
python Main.py CMC <Misclassification Type> <Misclassification Count>
```

### Arguments

- `<Misclassification Type>`:
  - `Any`     – Misclassify **any** datapoints (randomly selected)
  - `Correct` – Misclassify **only correctly classified** datapoints

- `<Misclassification Count>`:
  - Integer specifying how many datapoints to modify via MILP
## Add Your Own Dataset

### Image

To add a custom image dataset:

- Update the `GetDataset` function in `Utils/GetModelsDatasets.py`.
- You can refer to examples using datasets from `torchvision.datasets` as well as datasets loaded from local files.

### Tabular

- If your dataset is available on OpenML, simply pass the appropriate dataset name as an argument.
- To use a local file, see the example in the `LoadDataset` function in `Tabular/Main.py`.

---

## Add Your Own Architecture

### Image

- Modify the `Utils/CNNetworks.py` file to define your architecture.
- Also update the `GetModel` function in `Utils/GetModelsDatasets.py` to load your model.

### Tabular

- Modify the `Tabular/Networks.py` file to define your architecture.
- Also update the `TrainNN` function in `Tabular/Main.py` to incorporate your model.
