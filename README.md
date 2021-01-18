
# Graph Convolutional Networks (GCN) with vanilla Knowledge Distillation

  

This is the Pytorch implementation of Teacher-Student architecture (proposed by [[Hinton et al. 2015]](https://arxiv.org/abs/1503.02531)) using Graph Convolutional Network (GCN) as a base model. The GCN code is adpated from [Kipf's PyGCN](https://github.com/tkipf/pygcn).

  

Due to a lack of source codes of knowledge distillation relating to GCN, the vanilla implementation might be helpful for the research community.

  

### Dependencies

  

- Compatible with PyTorch 1.4.0 and Python 3.7.3.


### Environment

  

- The implementation is supposed to train in the GPU enviornment.

- We test all of the datasets on GeForce RTX 2080 Ti and CPU with 128GB RAM.

  

### Dataset:

  

- To fully fit the KD loss function, we run GCN with CORA, CITECEER, and PUBMED as benchmark datasets for semi-supervised node classification.

- CORA, CITECEER, and PUBMED are included in `data` directory. See details of dataset information in [[Sen et al. 2008]](https://ojs.aaai.org//index.php/aimagazine/article/view/2157)

  

### Result:

The belows are node classification results with ten differnet random initializations. It is noted that both teacher and student model have identical settings (2-layer GCN).

| acc(%) \ data | CORA | CITESEER | PUBMED |
|----|----|----|----|
| Teacher Model | 81.16±0.57 | 70.74±0.68 | 78.96±0.49 |
| Student Model | 81.83±1.53 | 71.45±0.85 | 79.48±0.41 |
  

### Training scripts (node classification):

  

It is noted that the hyper-parameters below have not been fine-tuned yet, but they are good enough for testing the efficiency of knowlede distillation on GCN.

- CORA:

  

```shell

python run.py --data cora --dropout 0.7 --T 4.0 --lambda_kd 0.7

```

  

- CITECEER:

  

```shell

python run.py --data citeceer --dropout 0.5 --T 3.0 --lambda_kd 0.7

```

  

- PUBMED:

  

```shell

python run.py --data pubmed --dropout 0.7 --T 6.0 --lambda_kd 0.8

```

  

Note: Results depend on random seed and will vary between re-runs.

  

-  `--seed` denotes the manual seed for model training

-  `--data` denotes training datasets

-  `--hidden` is the dimension of hidden GCN Layers

-  `--lr` denotes learning rate

-  `--l2` is the weight decay parameter of L2 regularization

-  `--dropout` is the dropout value for training GCN Layers

-  `--T` is the tempoerature for KD loss of soft targets

-  `--lambda_kd` is the hyper-parameter of KD loss.

- Rest of the arguments can be listed using `python run.py -h`
