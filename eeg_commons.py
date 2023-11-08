from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
# import leidenalg as la

from matplotlib import patches
from numpy import linalg
from scipy.io import loadmat
from scipy import signal
# from importlib import reload

'''
Main Function: FilteredSignals()

Returns a list of length 20 that contains the signals for each of the 20 subjects. 

All other functions are helper functions copied from other code. 


'''


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(PROJECT_DIR, "data", "real", "eeg")

electrode_positions = """
    {
        "Angle":{
            "FP1":-18.0,
            "AF7":-36.0,
            "AF3":-25.0,
            "F1":-22.0,
            "F3":-39.0,
            "F5":-49.0,
            "F7":-54.0,
            "FT7":-72.0,
            "FC5":-69.0,
            "FC3":-62.0,
            "FC1":-45.0,
            "C1":-90.0,
            "C3":-90.0,
            "C5":-90.0,
            "T7":-90.0,
            "TP7":252.0,
            "CP5":249.0,
            "CP3":242.0,
            "CP1":225.0,
            "P1":202.0,
            "P3":219.0,
            "P5":229.0,
            "P7":234.0,
            "P9":230.0,
            "PO7":216.0,
            "PO3":205.0,
            "O1":198.0,
            "IZ":180.0,
            "OZ":180.0,
            "POZ":180.0,
            "PZ":180.0,
            "CPZ":180.0,
            "FPZ":0.0,
            "FP2":18.0,
            "AF8":36.0,
            "AF4":25.0,
            "AFZ":0.0,
            "FZ":0.0,
            "F2":22.0,
            "F4":39.0,
            "F6":49.0,
            "F8":54.0,
            "FT8":72.0,
            "FC6":69.0,
            "FC4":62.0,
            "FC2":45.0,
            "FCZ":0.0,
            "CZ":90.0,
            "C2":90.0,
            "C4":90.0,
            "C6":90.0,
            "T8":90.0,
            "TP8":108.0,
            "CP6":111.0,
            "CP4":118.0,
            "CP2":135.0,
            "P2":158.0,
            "P4":141.0,
            "P6":131.0,
            "P8":126.0,
            "P10":130.0,
            "PO8":144.0,
            "PO4":155.0,
            "O2":162.0
        },
        "Radius":{
            "FP1":1.0,
            "AF7":1.0,
            "AF3":0.8043278085,
            "F1":0.5435082873,
            "F3":0.6521639042,
            "F5":0.8153775322,
            "F7":1.0,
            "FT7":1.0,
            "FC5":0.7826887661,
            "FC3":0.5435082873,
            "FC1":0.3478360958,
            "C1":0.25,
            "C3":0.5,
            "C5":0.75,
            "T7":1.0,
            "TP7":1.0,
            "CP5":0.7826887661,
            "CP3":0.5435082873,
            "CP1":0.3478360958,
            "P1":0.5435082873,
            "P3":0.6521639042,
            "P5":0.8153775322,
            "P7":1.0,
            "P9":1.25,
            "PO7":1.0,
            "PO3":0.8043278085,
            "O1":1.0,
            "IZ":1.25,
            "OZ":1.0,
            "POZ":0.75,
            "PZ":0.5,
            "CPZ":0.25,
            "FPZ":1.0,
            "FP2":1.0,
            "AF8":1.0,
            "AF4":0.8043278085,
            "AFZ":0.75,
            "FZ":0.5,
            "F2":0.5435082873,
            "F4":0.6521639042,
            "F6":0.8153775322,
            "F8":1.0,
            "FT8":1.0,
            "FC6":0.7826887661,
            "FC4":0.5435082873,
            "FC2":0.3478360958,
            "FCZ":0.25,
            "CZ":0.0,
            "C2":0.25,
            "C4":0.5,
            "C6":0.75,
            "T8":1.0,
            "TP8":1.0,
            "CP6":0.7826887661,
            "CP4":0.5435082873,
            "CP2":0.3478360958,
            "P2":0.5435082873,
            "P4":0.6521639042,
            "P6":0.8153775322,
            "P8":1.0,
            "P10":1.25,
            "PO8":1.0,
            "PO4":0.8043278085,
            "O2":1.0
        }
    }
""" 

def get_raw_fname(subject):
    fname =  f"{subject:03d}_L4_Er.mat"
    # return Path(DATA_DIR, "raw", fname)
    return Path("raw", fname)

def load_raw_data(n_subjects):
    raw_eeg = []
    min_trial_numbers = np.inf
    for s in range(n_subjects):
        fname = get_raw_fname(s+1)
        # Returns an array of size 64: each entry is n_trialsx512 matrix whose rows
        # are an electrode signal for a trial
        print(loadmat(fname))
        X = loadmat(fname)["data_elec"][0].squeeze() 

        raw_eeg.append(np.stack(X, axis=0))

        print(X[0].shape[0])

        min_trial_numbers = min(min_trial_numbers, X[0].shape[0])

    # Equate trial number across subjects
    for s in range(n_subjects):
        raw_eeg[s] = raw_eeg[s][:, :min_trial_numbers, :]

    return raw_eeg

def filter_signals(raw_eeg, band):
    filtered_eeg = []
    for s, X in enumerate(raw_eeg):
        b, a = signal.butter(N=4, Wn=band, btype="bandpass")
        filtered_eeg.append(signal.lfilter(b, a, X))

    return filtered_eeg

def sample_signals(filtered_eeg, window):
    return [X[:, :, window[0]:window[1]] for X in filtered_eeg]

def preprocess_data(n_subjects):
    raw_eeg = load_raw_data(n_subjects)
    filtered_eeg = filter_signals(raw_eeg, band=[0.0156, 0.0313])
    sampled_eeg = sample_signals(filtered_eeg, window=[103, 153])

    sampled_eeg = [np.transpose(X, (1, 0, 2)) for X in sampled_eeg]

    return sampled_eeg

def sample_trials(eeg_data, trials_to_use):
    return [X[trials_to_use, ...] for X in eeg_data]

def prep_data_for_gl(eeg_data):
    Xs = []
    for s, X_all in enumerate(eeg_data):
        n_trials, n_electrodes, _ = X_all.shape

        X_subject = []
        for t in range(n_trials):
            X_subject.append(X_all[t,:,:].squeeze())

        Xs.append(np.concatenate(X_subject, 1))

    return Xs

def get_graph_fname(method, frac_trials, density, similarity=None):
    fname = Path(DATA_DIR, "graphs", method, f"frac_trials_{frac_trials:.2f}", 
                 f"density_{density:.2f}")
    if similarity:
        fname = Path(fname, f"similarity_{similarity:.2f}")

    return Path(fname, "adjacencies.npz")

#################
# VISUALIZATION #
#################

def draw_head(ax, lw):
    # head
    patch = patches.Circle([0, 0], 1.25, facecolor="white", ec="0.5", zorder=-4, 
                           lw=lw)
    ax.add_patch(patch)
    
    # nose
    patch = patches.Ellipse((0, 1.25), 0.25, 0.3, ec="0.5", zorder=-5, lw=lw, 
                            fill=False)
    ax.add_patch(patch)

    # ears
    patch = patches.Ellipse((1.25, 0), 0.2, 0.4, ec="0.5", zorder=-5, lw=lw, 
                            fill=False)
    ax.add_patch(patch)
    patch = patches.Ellipse((-1.25, 0), 0.2, 0.4, ec="0.5", zorder=-5, lw=lw, 
                            fill=False)
    ax.add_patch(patch)

def plot_edges(edges: pd.DataFrame, ax: plt.Axes, electrode_size, font_size,
               electrode_colors="lightsteelblue"):
    # Load the position of electrodes
    electrode_pos = pd.read_json(electrode_positions, orient="columns")

    angles_in_radians = np.radians(electrode_pos["Angle"])
    electrode_pos["X"] = electrode_pos["Radius"]*np.sin(angles_in_radians)
    electrode_pos["Y"] = electrode_pos["Radius"]*np.cos(angles_in_radians)

    draw_head(ax, 1)

    if edges is not None:
        ew = edges.EdgeWeight.to_numpy().copy()
        ew -= np.min(ew)
        ew /= np.max(ew)

        i = -1
        for _, edge in edges.iterrows():
            i += 1
            xs = electrode_pos.loc[edge.Source, "X"]
            ys = electrode_pos.loc[edge.Source, "Y"]
            xt = electrode_pos.loc[edge.Target, "X"]
            yt = electrode_pos.loc[edge.Target, "Y"]

            ax.plot([xs, xt], [ys, yt], c='0.8', zorder=0, lw=ew[i])

    # Draw the electrodes
    ax.scatter(x=electrode_pos["X"], y=electrode_pos["Y"], s=electrode_size, 
                c=electrode_colors, linewidths=0.5, edgecolor="white", zorder=1)

    # Print electrode names
    for electrode, pos in electrode_pos.iterrows():

        x = pos["X"]
        y = pos["Y"] 

        # print the text
        ax.text(x = x, y = y, s=electrode, va='center', ha="center", 
                fontsize=font_size, zorder=2, color="black")

    # Adjusting plot limits
    stretch = 0.02
    xlim = ax.get_xlim()
    w = xlim[1] - xlim[0]
    ax.set_xlim([xlim[0]-stretch*w, xlim[1]+stretch*w])
    ylim = ax.get_ylim()
    h = ylim[1] - ylim[0]
    ax.set_ylim([ylim[0]-stretch*h, ylim[1]+stretch*h])

    # Ax setup
    ax.set(aspect='equal')
    ax.axis("off")

#########
# UTILS #
#########

def electrode_pairs():
    # Returns all electrode pairs as pandas DataFrame, which includes two
    # columns named source and target.

    electrodes = pd.read_json(electrode_positions)
    electrodes["Electrode"] = electrodes.index 
    electrodes.reset_index(inplace=True, drop=True)

    source = [i for i, j in combinations(electrodes.Electrode.to_list(), 2)]
    target = [j for i, j in combinations(electrodes.Electrode.to_list(), 2)]
    
    return pd.DataFrame({"Source": source, "Target": target})

def adj_to_pd(A):
    # Convert an adjancy matrix of EEG network to a pandas dataframe with 3 
    # columns: source, target, weight.
    
    ew = A[np.triu_indices_from(A, k=1)]

    out_pd = electrode_pairs()
    out_pd["EdgeWeight"] = ew
    out_pd = out_pd[out_pd.EdgeWeight != 0]

    return out_pd

#region COMMUNITY DETECTION

def find_comms(G, weights=None, resolution=1, n_runs=100):
    algo = lambda T: maximize_modularity(T, weights, resolution, n_runs)
    partition = find_consensus_comms(G, algo(G), algo, True).astype(int)

    electrodes = pd.read_json(electrode_positions)
    return pd.DataFrame({
        "Electrode": electrodes.index,
        "Community": partition
    })

def maximize_modularity(G, weights=None, res=1, n_runs=1,):
    partitions = np.zeros((G.vcount(), n_runs))
    
    for r in range(n_runs):
        C = la.find_partition(G, la.RBConfigurationVertexPartition, 
                              resolution_parameter=res, weights=weights, 
                              seed=r+10)
        partitions[:, r] = C.membership

    return partitions

def association_matrix(partitions, th):
    n_nodes, n_partition = partitions.shape

    A = np.zeros((n_nodes, n_nodes))
    for p in range(n_partition):
        A += (partitions[:, p][..., None] == partitions[:, p]).astype(float)/n_partition

    if th:
        # Randomize partitions and get a randomized association matrix
        for p in range(n_partition):
            np.random.shuffle(partitions[:, p])

        A_rnd = np.zeros((n_nodes, n_nodes))
        for p in range(n_partition):
            A_rnd += (partitions[:, p][..., None] == partitions[:, p]).astype(float)/n_partition

        # Expected number of times a pair of nodes is assigned to the same community in random partition
        threshold = np.max(A_rnd[np.triu_indices(n_nodes, k=1)])

        A_cp = A.copy()
        A[A<threshold] = 0
        A[np.diag_indices(n_nodes)] = 0

        # After thresholding some nodes may be disconnected, connect them to neighbors with high weights  
        degrees = np.count_nonzero(A, axis=1)
        for i in np.where(degrees==0)[0]:
            for j in np.where(A_cp[i, :] >= np.max(A_cp[i, :])-1e-6)[0]:
                A[i, j] = A_cp[i, j]

    return A

def find_consensus_comms(G, partitions, alg, n_calls=1, th=False):
    # Association matrix
    A_org = association_matrix(partitions.copy(), th)

    # When partitions are the same across runs, association matrix includes only two values
    # We can use this fact to stop the algorithm. 
    if len(np.unique(A_org)) == 2 or n_calls>20:
        return partitions[:, 0]

    # construct an igraph from association matrix
    T = ig.Graph.Weighted_Adjacency(A_org, mode="undirected", loops=False)

    for attr in G.vertex_attributes():
        T.vs[attr] = G.vs[attr]

    return find_consensus_comms(G, alg(T), alg, n_calls=n_calls+1)



def FilteredSignals():

    raw_eeg = load_raw_data(20)
    filtered_eeg = filter_signals(raw_eeg, band=[0.0156, 0.0313])
    sampled_eeg = sample_signals(filtered_eeg, [103, 153])
    
    W1Average = np.mean(sampled_eeg, axis=2)
    return W1Average

