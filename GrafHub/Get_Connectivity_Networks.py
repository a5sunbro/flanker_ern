import os
import glob
import igraph as ig
import gzip
import shutil

from scipy import sparse
from pathlib import Path

import numpy as np

'''
Main Function: GraphAdjacencyMatrix(subject)

GraphAdjacencyMatrix(subject) gets the functional connectivity netowks. 

All other files are helper files that were copied from previous code. 

'''


def read_gml(file_name):
    """Read a graph from a GML file.

    Parameters
    ----------
    file_name : str
        File to read. The file can also be gzipped file (ends with .gz).

    Returns
    -------
    G : igraph.Graph
        Read graph.
    """

    # If a Path object is provided as file, convert it to string
    if isinstance(file_name, Path):
        file_name = str(file_name)

    is_zipped = file_name.endswith(".gz")

    # create a temporary gml file to be able to read with igraph
    if is_zipped:
        with gzip.open(file_name, 'rb') as f_in:
            file_name = file_name[:-3]
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    G = ig.read(file_name, format="gml") # read the graph

    if is_zipped:
        os.remove(file_name) # remove the temporary gml file

    return G 

def S_layer(layer, layers):
    if layer is not None:
        if layer not in layers:
            raise Exception("{} is not a valid layer.".format(layer))

def S_edge_attribute(attr, attrs):
    if attr is not None:
        if attr not in attrs:
            raise ValueError("{} is not a valid edge attribute.".format(attr))
        
def S_layer_pairs(layer1, layer2):
    if (layer1 is None) and (layer2 is not None):
        raise Exception("Paremeter layer1 cannot be None, when layer2 is given.")

class MLGraph():
    """A composite class based on igraph for multilayer graph analysis.
    """

    def __init__(self, graph: ig.Graph=None) -> None:
        """Initiate either an empty multilayer graph or a multilayer graph from
        a given igraph Graph.

        Parameters
        ----------
        graph : igraph Graph, optional
            If provided, the multilayer graph will be generated from it. Vertices
            should have an attribute named "layer". By default None.

        Raises
        ------
        Exception
            If provided graph's vertices don't have attribute "layer".
        """
        if graph is None: # Create an empty graph
            self.graph = ig.Graph()
            self.layers = []
        else:
            if not ("layer" in graph.vertex_attributes()):
                raise Exception("Provided graph must have vertex attribute 'layer'.")
            else:
                self.graph = graph
                node_layers = np.array(self.graph.vs["layer"])
                _, indx = np.unique(node_layers, return_index=True)
                self.layers = list(node_layers[np.sort(indx)])

    def order(self, layer=None):
        """Return the number of nodes in a multilayer graph or in a given layer.

        Parameters
        ----------
        layer : str, optional
            Name of the layer whose node number will be returned, by default `None`.

        Returns
        -------
        order: int
            Number of nodes.
        """

        # Input check
        S_layer(layer, self.layers)
        
        # Return number of nodes in the multilayer graph
        if layer is None:
            return self.graph.vcount()

        # Return number of nodes in the given layer
        else:
            return len(self.layer_vertices(layer))

    def size(self, weight=None, layer1=None, layer2=None):
        """Return number of edges in a multilayer graph. If the parameter `weight`
        is not `None`, return total weight of edges.

        Parameters
        ----------
        weight : str, optional
            Edge attribute to use as edge weight, by default `None`.
        layer1 : str, optional
            If not `None` and `layer2` is `None`, consider only intralayer edges
            in `layer1`, by default `None`.
        layer2 : str, optional
            If not `None`, consider only interlayer edges between `layer1` and 
            `layer2`. The parameter `layer1` must be provided, if not `None`.
            By default `None`.

        Returns
        -------
        size: float, or int
            Number of edges or total edge weight if weight is not `None`.
        """
        
        # Input check
        S_edge_attribute(weight, self.graph.edge_attributes())
        S_layer(layer1, self.layers)
        S_layer(layer2, self.layers)
        S_layer_pairs(layer1, layer2)

        # Return the size of multilayer graph
        if layer1 is None and layer2 is None:
            size = sum([1 if weight is None else e["weight"] for e in self.graph.es])

        # Return the size of intralayer graph of layer1
        elif layer2 is None:
            # nodes in the given layer
            layer_nodes = self.layer_vertices(layer1)
            edges = self.graph.es.select(_within = layer_nodes)
            
            size = sum([1 if weight is None else e["weight"] for e in edges])

        # Return the size of interlayer graph between layer1 and layer2
        else:
            # nodes in the given layers
            layer1_nodes = self.layer_vertices(layer1)
            layer2_nodes = self.layer_vertices(layer2)
            edges = self.graph.es.select(_between = (layer1_nodes, layer2_nodes))
            
            size = sum([1 if weight is None else e["weight"] for e in edges])

        return size

    def layer_vertices(self, layer):
        """Return the set of nodes in a given layer.

        Parameters
        ----------
        layer : str
            Layer name.

        Returns
        -------
        nodes: ig.VertexSeq
            Nodes in the given layer.
        """

        S_layer(layer, self.layers)

        return self.graph.vs.select(layer_eq=layer)

    def degree(self, nodes, weight=None, layer=None):
        """Return (weighted, layer-wise) degrees of a set of nodes.

        Parameters
        ----------
        nodes : int, list of ints, or ig.VertexSeq
            A single node ID or a list of node ID.
        weight : str, optional
            Edge attribute to use as weight. If None regular degree will be 
            returned, by default None.
        layer : str, optional
            Name of a layer. If not None, return layer-wise degree of the nodes, 
            by default None.

        Returns
        -------
        degrees: float, or list of floats
            Node degrees.
        """

        S_edge_attribute(weight, self.graph.edge_attributes())
        S_layer(layer, self.layers)

        # If a vertex sequence is provided convert it to the list node ids
        if isinstance(nodes, ig.VertexSeq):
            nodes = nodes.indices

        # If a single vertex is given convert it to a list        
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Return total (weighted) degree of the node
        if layer is None:
            return self.graph.strength(nodes, weights=weight, loops=False)
        
        # Return layer-wise (weighted) degrees of nodes
        else:
            # nodes in the given layer
            layer_nodes = self.layer_vertices(layer).indices

            # Use adjacency matrix to get node strengths
            adj = self.graph.get_adjacency_sparse(attribute=weight)
            return np.sum(adj[nodes, :][:, layer_nodes], axis=1)

    def intralayer_graph(self, layer):
        """Return intralayer graph of a given layer.

        Parameters
        ----------
        layer : str
            The layer whose intralayer graph will be returned.

        Returns
        -------
        G : ig.Graph
            Intralayer graph.
        """
        S_layer(layer, self.layers)

        nodes = self.graph.vs(layer_eq=layer)
        return self.graph.induced_subgraph(nodes)

    def interlayer_graph(self, layer1, layer2):
        """Return interlayer graph that is between two given layers.  

        Parameters
        ----------
        layer1 : str
            Name of the first layer.
        layer2 : str, optional
            Name of the second layer.

        Returns
        -------
        G: ig.Graph
            Interlayer graph.
        """

        S_layer(layer1, self.layers)
        S_layer(layer2, self.layers)

        layer1_nodes = self.graph.vs(layer_eq=layer1)
        layer2_nodes = self.graph.vs(layer_eq=layer2)
        subgraph = self.graph.induced_subgraph(
            layer1_nodes.indices + layer2_nodes.indices
        )

        layer1_nodes = subgraph.vs(layer_eq=layer1)
        layer2_nodes = subgraph.vs(layer_eq=layer2)
        edges = subgraph.es(_within=layer1_nodes)
        subgraph.delete_edges(edges)
        edges = subgraph.es(_within=layer2_nodes)
        subgraph.delete_edges(edges)

        return subgraph

    def intralayer_adjacency(self, layer, weight=None):
        """Return adjacency matrix of a given layer.

        Parameters
        ----------
        layer : str
            The layer whose adjacency will be returned.
        weight : str, optional
            Attribute to use for edge weight, by default None

        Returns
        -------
        adj_mat : sparse matrix
            The adjacency matrix.
        """

        intra_graph = self.intralayer_graph(layer)

        # scipy sparse matrix gives error when graph is empty, 
        # so we handle it by ourselves
        if intra_graph.ecount() == 0:
            n_nodes = intra_graph.vcount()
            return sparse.csr_matrix((n_nodes, n_nodes))
        else:
            return intra_graph.get_adjacency_sparse(attribute=weight)

    def interlayer_incidence(self, layer1, layer2, weight=None):
        """Return the incidence matrix between two layers.

        Parameters
        ----------
        layer1 : str
            Name of the first layer.
        layer2 : str, optional
            Name of the second layer.
        weight : str, optional
            Attribute to use for edge weight, by default None

        Returns
        -------
        inc_mat : sparse matrix
            The incidence matrix.
        """
        inter_graph = self.interlayer_graph(layer1, layer2)

        # scipy sparse matrix gives error when graph is empty, 
        # so we handle it by ourselves
        if inter_graph.ecount() == 0:
            n_nodes1 = self.order(layer1)
            n_nodes2 = self.order(layer2)
            return sparse.csr_matrix((n_nodes1, n_nodes2))

        inter_graph_adj = inter_graph.get_adjacency_sparse(attribute=weight)

        layer1_nodes = inter_graph.vs(layer_eq=layer1).indices
        layer2_nodes = inter_graph.vs(layer_eq=layer2).indices

        return inter_graph_adj[layer1_nodes, :][:, layer2_nodes]

    def supra_adjacency(self, weight: str=None) -> dict:
        """Return the supra-adjacency of the multilayer network as dict-of-dict.

        Parameters
        ----------
        weight : str, optional
            Attribute to use for edge weight, by default None

        Returns
        -------
        supra: dict-of-dict
            Supra-adjacency matrix as dict-of-dict. supra[i][j] is the incidence
            matrix between layer i and j if i != j, and the adjacency matrix of
            layer i if i == j. 
        """
        layers = self.layers
        
        supra = {}

        for li in layers:
            supra[li] = {}
            for lj in layers:
                if li == lj:
                    supra[li][li] = self.intralayer_adjacency(li, weight)
                else:
                    supra[li][lj] = self.interlayer_incidence(li, lj, weight)

        return supra
                     
    def read_from_gml(self, file_name: str) -> None:
        """Read a multilayer graph from a (zipped) gml file. 

        Parameters
        ----------
        file_name : str
            The GML file to read. It can also be a zipped gml. 

        Raises
        ------
        Exception
            The vertices should have attribute "layer".
        """
        self.graph = read_gml(file_name)

        if "layer" not in self.graph.vertex_attributes():
            raise Exception("Graph read is not a multilayer graph. " +\
                            "Make sure vertices have 'layer' attribute.")

        node_layers = np.array(self.graph.vs["layer"])
        _, indx = np.unique(node_layers, return_index=True)
        self.layers = list(node_layers[np.sort(indx)])

    def write_to_gml(self, file_name: str) -> None:
        """Write the multilayer graph to a (zipped) GML file.

        Parameters
        ----------
        file_name : str
            Save name. It should include the extension. If it ends with .gz, the file
            is gzipped.
        """
        io.write_gml(self.graph, file_name)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(PROJECT_DIR)

def load_subject_graph(group, subject, layer=None):
    # Multilayer graph file
    file = Path(
        DATA_DIR, "eeg_reu", "Subjects", f"subject_{subject}.gml.gz"
    )
    
    # Read the graph
    graph = MLGraph()
    graph.read_from_gml(file)

    if layer :
        graph = graph.intralayer_graph(layer)
    
    return graph

def load_graphs(group):
    gml_files = Path(DATA_DIR, group, "networks", "*.gml")
    zipped_gml_files = Path(DATA_DIR, group, "networks", "*.gml.gz")
    network_files = (glob.glob(str(gml_files.absolute())) + 
                     glob.glob(str(zipped_gml_files.absolute())))

    graphs = {}
    for file in network_files:
        file_name = file.split(os.path.sep)[-1].split(".")[0]
        graphs[file_name] = MLGraph()
        graphs[file_name].read_from_gml(file)

    return graphs



def GraphAdjacencyMatrix(subject):
    #Load Data
    G = load_subject_graph("data", subject)
    #Get rid of non-Theta Nodes
    Theta = G.graph.vs.select(lambda y: y['layer']!="theta")
    G.graph.delete_vertices(Theta)
    #Get adhacenct matrix
    N = G.graph.get_adjacency_sparse(attribute = "weight").todense()
    #Format adjasceny matrix
    N = np.asarray(N.tolist())
    return N

