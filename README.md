# DyGETViz

## Installation


```bash
conda install scikit-learn pandas numpy matplotlib plotly
conda install -c conda-forge umap
pip install 
```


Please refer to the homepage of [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html) to install these 3 packages, respectively. 



## Getting Started


- highlighted_nodes: List of nodes to be highlighted in the visualization. We need to specify these nodes because we only show the names of a small number of nodes in the plotly visualization. Otherwise, the generated plot will be too messy. 


- **plot_dtdg.py**: Script for generating the visualization
- 

Generate the visualization using the command:

```bash
python dygetviz/plot_dtdg.py --dataset_name Chickenpox --model GConvGRU
```

## Note



- The Reddit dataset is a bit special because it is the only dataset that describes a bipartite graph. The first 60 snapshots are for each of the 60 snapshots. The last snapshot is for the background nodes. The shape of the embeddings is `` 

## Acknowledgments

We thank members of the CLAWS Lab and SRI International for their feedback and support.


