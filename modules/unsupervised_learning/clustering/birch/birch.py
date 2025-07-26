import csv
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from clustering_feature import ClusteringFeature
from cf_tree import CFTree


class BIRCH:
    def __init__(self, data_dim=2, page_size=512, threshold=0.0, max_size=100_000, estimated_data_size=105_000) -> None:
        self.max_size = max_size
        self.data_dim = data_dim
        self.branching_factor = page_size // sys.getsizeof(
            ClusteringFeature(data_dim))
        self.threshold = threshold
        self.max_size = max_size
        self.tree = CFTree(
            self.data_dim, self.branching_factor, self.threshold, True)
        self.old_thresholds: list[list[float]] = [[threshold]]
        self.N_added_list: list[list[int]] = []
        self.radii: list[list[float]] = []
        self.N = estimated_data_size
        self.outlier_threshold = 0
        self.outliers: list[ClusteringFeature] = []
        self.outlier_mem_size = 20 * max_size // 100

    def addall(self, nodefile):
        with open(nodefile, "rt") as f:
            reader = csv.reader(f)
            for row, text in enumerate(reader):
                # Controllo ogni 1000 righe che l'albero non sia troppo grande in MB
                if sys.getsizeof(self.tree) > self.max_size:
                    # Ricalcola threshold
                    threshold = self.recompute_threshold(row)
                    # Comprimi l'albero in uno più piccolo usando la nuova soglia
                    self.compress(threshold)
                # Aggiungo un nodo
                coordinates = np.array(text, dtype=float)
                cf = ClusteringFeature(self.data_dim).add_point(coordinates)
                self.tree.insert_CF(cf)
                # Controlla che la root non sia troppo grande
                if len(self.tree.CF_list) > self.branching_factor:
                    # Crea una nuova
                    new_root = CFTree(
                        self.data_dim, self.branching_factor, self.threshold, False)
                    self.tree.parent = new_root
                    new_root.CF_list.append(self.tree.compute_cumulative_CF())
                    new_root.children_list.append(self.tree)
                    # Dividi quella vecchia
                    new_root.split_child(0)
                    self.tree = new_root
            # Check se gli outliers possono essere assorbiti back nell'albero, altrimenti vengono sartati
            self.compress(self.threshold)
            to_remove = [self.tree.insert_CF(
                outlier, outlier=True) for outlier in self.outliers]
            self.outliers = [outlier for idx, outlier in enumerate(
                self.outliers) if not to_remove[idx]]
            return self

    def recompute_threshold(self, row: int):
        # Stima quanti dati vogliamo aggiungere alla prossima iterazione
        N_next = min(self.N, 2 * row)
        # Calcola il raggio della root
        root = self.tree.compute_cumulative_CF()
        radius = root.radius()
        self.radii.append([radius])
        self.N_added_list.append([row])

        model = LinearRegression()
        # Packed volume V_p = T^d, aumenta linearmente con con la threshold
        N_power = [[number[0] ** (1/self.data_dim)]
                   for number in self.N_added_list]
        # Stima la prossima threshold
        model.fit(N_power, self.old_thresholds)
        next_threshold = model.predict(np.array(N_next).reshape(-1, 1))[0][0]
        # Stima il prossimo raggio
        model.fit(self.N_added_list, self.radii)
        radius = model.predict(np.array(N_next).reshape(-1, 1))[0][0]

        expansion_factor = max(1.0, float(radius / self.radii[-1][0]))
        # Trova nel cluster con più nodi le foglie più vicine
        d_min = self.tree._find_smallest_increase()

        next_threshold = min(d_min, float(expansion_factor * next_threshold))
        # Se non è abbastanza grande incrementala
        if next_threshold <= self.old_thresholds[-1][0]:
            next_threshold = next_threshold*(N_next/row)**(1/self.data_dim)
        # Evita 0.0 come nuova threshold
        if next_threshold == 0.0:
            next_threshold = 0.0001
        self.old_thresholds.append([next_threshold])
        return next_threshold

    def compress(self, threshold):
        self.threshold = threshold
        # Soglia per decidere se un leaf CF è un outlier
        # Calcola il numero medio di nodi nelle foglie
        self.outlier_threshold = max(
            self.outlier_threshold, self.tree._avg_nodes_in_leaves() // 4)
        compressed = CFTree(
            self.data_dim, self.branching_factor, self.threshold, True)

        for leaf in self.tree.leaves():
            if leaf.N > self.outlier_threshold:
                compressed.insert_CF(leaf)
            # Se questo nodo ha meno di 20% di punti della media
            else:
                self.manage_outliers(compressed, leaf)
        self.tree = compressed
        return self

    def manage_outliers(self, compressed: CFTree, leaf: ClusteringFeature):
        # Aggiungi alla lista dioutliers
        self.outliers.append(leaf)
        # Se gli outlier sono troppi
        if sys.getsizeof(self.outliers) > self.outlier_mem_size:
            # Reinserisci outlier e guarda quali sono stati reinseriti
            to_remove = [compressed.insert_CF(
                outlier, outlier=True) for outlier in self.outliers]
            # Tieni quelli che non sono stati reinseriti
            self.outliers = [outlier for idx, outlier in enumerate(
                self.outliers) if not to_remove[idx]]
