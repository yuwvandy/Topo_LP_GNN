# Topo-LP-GNN
This repository is an official PyTorch(Geometric) implementation of analysis/models in ["A Topological Perspective on Demystifying GNN-Based Link Prediction Performance"]().

**If you use this code, please consider citing:**
```linux

```

## Motivation
This paper aims to study the varying GNNs' link prediction performance across nodes within a graph. Specifically, we aim to propose a node-level topological metric, Topological Concentration, and demonstrate its superior correlation with LP performance to other node topological properties like degree/local subgraph density.

![](./img/analysis.png)

## Analysis
We perform comprehensive analysis including correlation analysis at node/graph levels, cold-start analysis and distribution shift analysis.  
See [[Analysis]](./Analysis/.) for more details.

## Model
We propose topological reweight to enhance node TC and empirically demonstrate its effectiveness in LP.  
See [[Model]](./Model/.) for more details.
