


GNNProject: Graph-Structured Inductive Bias in Deep Learning
=========================================================================
.. raw:: html

We investigate to which extent inferring the graph structure from the data and using it as an inductive bias to a Graph Neural Network (GNN) improves robustness and generalization in comparison to the standard fully connected neural networks (FCNN) without any prior structural information. Furthermore, we explore how the quality of an inductive bias (i.e., the graph structure reconstructed from the high-dimentional data) impacts the performance. To that end, we carry out multiple experiments with both synthetic and real datasets, to compare performance of GNNs and FCNNs in various settings, varying characteristics of the input dataset, such as number of features, number of observations, noise level, or quality of the input structure. 

The models are implemented with `pytorch <https://pytorch.org/docs/stable/index.html>`_ and `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ in python.

**NOTE:** Here, we talk about graph classification, where each observation to be classified is represented by a graph, with nodes representing features. The graph structure for all observations is the same, the only difference comes from node values.


Modules
-------------------------------

Synthetic data generation
**********************
We generate graph structured, labeled, synthetic data inspired by biological networks. It consists of two steps:
1. Random Graph Generation:

- **Erdős–Rényi model:** To generate plane random networks based on `Erdős & Rényi, 1959 <https://en.wikipedia.org/wiki/Barabási–Albert_model>`_)

- **Barabási–Albert model:** To generate "scale-free" networks based on `Albert & Barabási, 2002 <https://en.wikipedia.org/wiki/Barabási–Albert_model>`_).

- **Stochastic Block model:** To generate "modular" networks based on `Stochastic Block Models <https://en.wikipedia.org/wiki/Stochastic_block_model>`_)

2. Message Passing on Graph (Data Generation):
This happens once the graph is generated and the nodes are initialized by a Gaussian white noise. For each class a subset of nodes (features) are chosen randomly as 'characteristics'. After message passing, another Gaussian white noise is added to all nodes.  

- **Information diffusion:** A Gaussian signal with a non-zero mean is added initially to the characteristic nodes. Each edge passes information based on the (global) edge capacity (diffusion coefficient) and the difference between its end nodes. Such a diffusion can be performed multiple times.

- **Activation:** Characteristic nodes are 'activated' with a Gaussian signal weighted by average of their neighbors. 

- **Sign:** Sign of characteristic nodes is set based on the sign of average of their neighbors. 

Real data
**********************
Our framework is applicable to any kind of (n_obs * n_features) multi-class dataset, as it is able to reconstruct the underlying graph. Indeed, the performance is better when there is an underlying dependence structure among features. A sample real dataset can be found `here <https://polybox.ethz.ch/index.php/s/12DdfFYADCetsNE>`_. This is a preprocessed version of `this dataset <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044>`_.

Graph inference
**********************
We infer the underlying graph structure based on the sparse high-dimensional correlation matrices estimated using the graphical lasso (`Friedman et al., 2008 <https://en.wikipedia.org/wiki/Graphical_lasso>`_) or the Ledoit-Wolf shrinkage estimator (`Ledoit & Wolf, 2004 <https://en.wikipedia.org/wiki/Graphical_lasso>`_). The adjacency matrix for the observed dataset and correspondingly the graph structure is reconstructed based on estimation of the inverse of a sparse high-dimensional correlation matrix. 

Classifiers
**********************
We use GNNs and compare them with FCNNs as baselines. We define GNNs based on `message passing <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html>`_: each node generates a message based on its features, and sends it to its neighbors. Then each node aggregates the messages of its neighbors and uses this aggregate to updates its features. The aggregation is done with a permutation-invariant function, like mean, sum or max. Typically, we use one round of message passing and we use 8 hidden features per node. Several types of graph convolutional layers are implemented: 

 - `GraphSAGE <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv>`_ 
 - `Chebyshev convolutions <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv>`_
 - `MFConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.MFConv>`_
 - `GraphConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv>`_ 
 - `GIN <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv>`_ 
 - `GATConv (graph attention) <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv>`_
 - `TransformerConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv>`_


Usage and installation
-------------------------------
You can get the latest development version of our toolkit from `Github <https://github.com/e-sollier/DL2020/>`_ using the following steps:
First, clone the repository using ``git``::

    git clone https://github.com/e-sollier/DL2020

Then, ``cd`` to the scArches folder and run the install command::

    cd DL2020
    python setup.py install

If you have problems with the torch-geometric dependency, look at `this <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_. 

Dependencies on ETH's Leonhard Cluster
**********************
In order to solve dependencies on Leonhard one should take the following steps:
1. Import the following modules::

    module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

2. Install relevant packages for torch-geometric::
    
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-geometric


Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/e-sollier/DL2020/issues/new>`__ or reach us by `email <mailto:eheidari@student.ethz.ch, esollier@student.ethz.ch, azagidull@student.ethz.ch>`_.

Reference
-------------------------------
The extend version of report for this project can be found `here <https://polybox.ethz.ch/index.php/s/FYnQKXRfeWoHlqO>`_.

Reproducing the report figures 
**********************
- Fig. 1: `Experiments/run_batch_graphQual.sh` --> `Experiments/read_results_graphQual.sh`
- Fig. 2: `Experiments/run_batch_obs.sh` --> `Experiments/read_results_obs.sh`
- Fig. 3: `Experiments/run_batch_features.sh` --> `Experiments/read_results_features.sh`
- Fig. 4: `Experiments/run_batch_real.sh` --> `Experiments/read_results_real.sh`
- Fig. 5: `Experiments/run_batch_noise.sh` --> `Experiments/read_results_noise.sh`
- Fig. 6: `Experiments/run_batch_layers.sh` --> `Experiments/read_results_layers.sh`
- Fig. 7: `Experiments/run_batch_alpha.sh` --> `Experiments/read_results_alpha.sh`



