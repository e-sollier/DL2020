from sklearn.covariance import GraphicalLassoCV

def get_adjacency_matrix_GL(data):
    
    cov = GraphicalLassoCV().fit(data)
    precision_matrix = cov.get_precision()
    adjacency_matrix = precision_matrix.astype(bool).astype(int)
    
    return adjacency_matrix

