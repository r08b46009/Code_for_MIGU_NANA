# Installation
This is the code for NaNa and MiGu: Semantic Data Augmentation Techniques to Enhance Protein Classification in Graph Neural Networks

Required environment: Python 3.8, PyTorch 2.0.1

For MIGU and NANA algorithm: 

```
conda activate MIGU conda env update --file environment_pygdemo.yml
```

For generating biophysical & other features: 

```
conda activate hydrogen_bond conda env update --file environment_bio.yml
```

# Dataset Preparation

Please download the dataset XXX from *** and put the file under the folder XXX

# Run Augmentation

## Generate Augmented Dataset

Please run the following command

```
python hydrogen_bond.py
```

## Train the Model

### Command Line Options

- ``--training_title``:  Name of the training, type: string
- ``--epochs``: Training epochs, type: float 
- ``--lr``: Learning rate, type: Float
- ``--optimizer``: Name of the optimizers, type: string, options: ``[adam]``
- ``--dim_h``: the width of the layers, type: integer
- ``--batch_size``: Training batch size, type: 
- ``--model``: Model types, type: string, options: ``[GIN_Attribute]``
- ``--num_workers``: Number of dataloader workers, type: integer
- ``--dataset_path``: The path of loaded dataset, type: string
- ``--dataset``: The used dataset for the training, type: string
- ``--edge``: Whether to use co-embedding residual learning, type: boolean
- ``--eval_batch_size``: The batch size in the evaluation stage, type: integer
- ``--version``: Use augmented or non-augmented data, "mine" means augmented data, "their" means non-augmented data, type: string, options: ``[their, mine]``
- ``--type``: The types of layers, type: string, options: ``[GCN, GIN, MPNN]``
- ``--bond``: Use edge attributes or not; if the flag is true, it would represent the MiGu augmentation; otherwise it is NaNa augmentation, type: boolean

### Run with the Following Command

For example, to train a GCN model with augmented EC dataset and residual learning framework, you can use the following command

```
CUDA_LAUNCH_BLOCKING=1 python graph_hao_retry.py --training_title 226_MPNN_true_edge --epochs 4000 --lr 0.0005 --optimizer adam --dim_h 128 --batch_size 50 --model GIN_Attribute --num_workers 8 --dataset_path /home/ysl_0128/DIG/examples/threedgraph/dataset --dataset EC --edge true --eval_batch_size 20 --version mine --type GCN --bond false
```
