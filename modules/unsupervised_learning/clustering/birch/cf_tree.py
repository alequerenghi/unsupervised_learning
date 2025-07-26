import copy
import sys
import numpy as np
from clustering_feature import ClusteringFeature


class CFTree:
    def __init__(self, data_dimensionality, branching_factor, threshold, is_leaf: bool, parent=None):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.data_dimensionality = data_dimensionality
        self.parent = parent
        self.CF_list: list[ClusteringFeature] = []
        # Lista dei figli (solo se non è foglia)
        self.children_list: list['CFTree'] = []

    def insert_CF(self, cf: ClusteringFeature, outlier=False) -> bool:
        # Se nodo leaf
        if self.is_leaf:
            # Se non ci sono ancora CF nel nodo leaf
            if not self.CF_list:
                self.CF_list.append(cf)
                return True

            # Provo a vedere se il più vicino e quello in input sono < T
            # 'closest_cf' è un CF, il più vicino al CF passato in input
            closest = self.find_closest_CF(cf)
            closest_cf = self.CF_list[closest]
            temp_cf = copy.copy(closest_cf)
            temp_cf.merge(cf)

            if temp_cf.radius() < self.threshold:
                # Se si, we are good to go, aggiungo e basta
                closest_cf.merge(cf)
                return True
            elif not outlier:
                # Altrimenti aggiungo il CF al nodo
                self.CF_list.append(cf)
                return True
        # Se il nodo non è foglia
        else:
            closest = self.find_closest_CF(cf)
            # Attraverso l'indice trovo il nodo figlio
            child = self.children_list[closest]

            # Inserisco il CF nel figlio, aka chiamo ricorsivamente questo metodo
            to_return = child.insert_CF(cf)

            if len(child.CF_list) > self.branching_factor:
                seed1, seed2 = self.split_child(closest)
                if len(self.children_list) <= self.branching_factor:
                    # Merging refinement
                    self.merge_refinement(seed1, seed2)
            else:
                # Aggiorna il CF cumulativo del figlio
                self.CF_list[closest] = self.children_list[closest].compute_cumulative_CF(
                )
            return to_return
        return False

    def _find_index(self, mode="furthest"):
        # Trova l'indice della coppia di CF più lontani o più vicini
        op = 1.0 if mode == "furthest" else -1.0
        seed1 = 0
        seed2 = 0
        max_dist = -float("inf")
        for i in range(len(self.CF_list)-1):
            for j in range(i + 1, len(self.CF_list)):
                dist = op * self.CF_list[i].dist(self.CF_list[j])
                if dist > max_dist:
                    max_dist = dist
                    seed1 = i
                    seed2 = j
        return seed1, seed2

    def split_child(self, idx: int) -> None:
        # Esegue lo split del figlio all’indice `idx` nel nodo corrente
        child = self.children_list[idx]
        # Find i nodi più lontani
        seed1, seed2 = child._find_index(mode="furthest")

        # Crea due nuovi nodi
        params_dict = {"data_dimensionality": self.data_dimensionality,
                       "branching_factor": self.branching_factor,
                       "threshold": self.threshold,
                       "is_leaf": child.is_leaf,
                       "parent": self
                       }
        new_node1 = CFTree(**params_dict)
        new_node2 = CFTree(**params_dict)
        # Inserisce i due semi nei nuovi nodi
        new_node1.CF_list.append(child.CF_list[seed1])
        new_node2.CF_list.append(child.CF_list[seed2])
        if not child.is_leaf:
            # >> hanno figli
            new_node1.children_list.append(child.children_list[seed1])
            new_node2.children_list.append(child.children_list[seed2])

        # Inserisco i restanti CF
        for node_idx, cf in enumerate(child.CF_list):
            if node_idx not in (seed2, seed1):
                dist1 = cf.dist(new_node1.CF_list[0])
                dist2 = cf.dist(new_node2.CF_list[0])
                if dist1 < dist2:
                    # Inserisco nel nuovo tree
                    new_node1.CF_list.append(cf)
                    if not child.is_leaf:
                        # >> ha figli e li inserisco
                        new_node1.children_list.append(
                            child.children_list[node_idx])
                else:
                    new_node2.CF_list.append(cf)
                    if not child.is_leaf:
                        new_node2.children_list.append(
                            child.children_list[node_idx])
        # Aggiunge i due nuovi nodi come figli del nodo corrente
        self.children_list.append(new_node1)
        self.CF_list.append(new_node1.compute_cumulative_CF())
        self.children_list.append(new_node2)
        self.CF_list.append(new_node2.compute_cumulative_CF())
        # Rimuove il nodo figlio originale
        self.CF_list.pop(idx)
        self.children_list.pop(idx)
        # Corregge gli indici se necessario
        if seed1 > idx:
            seed1 -= 1
        if seed2 > idx:
            seed2 -= 1
        return None

    def find_closest_CF(self, cf: ClusteringFeature) -> int:
        # Trova l'indice del CF più vicino in base al centroide
        min_dist = float('inf')
        closest = 0
        for idx, node in enumerate(self.CF_list):
            dist = np.linalg.norm(node.centroid() - cf.centroid())
            if dist < min_dist:
                min_dist = dist
                closest = idx
        return closest

    def compute_cumulative_CF(self):
        # Calcola il ClusteringFeature aggregato del nodo
        cf = ClusteringFeature(self.data_dimensionality)
        for entry in self.CF_list:
            cf.merge(entry)
        return cf

    def _find_smallest_increase(self) -> float:
        # Calcola la distanza minima tra coppie per stimare densità
        if self.is_leaf:
            # Trova i più vicini
            seed1, seed2 = self._find_index("smallest")
            dist = self.CF_list[seed1].dist(self.CF_list[seed2])
            # Ritorna la loro distanza
            return dist
        # Else trova il figlio più denso
        idx_max = np.argmax(list(map(lambda x: x.N, self.CF_list)))
        return self.children_list[idx_max]._find_smallest_increase()

    def merge_refinement(self):
        # Step 1: trova i due CF più vicini
        seed1, seed2 = self._find_index("closest")
        split1 = len(self.children_list) - 2
        split2 = len(self.children_list) - 2
        # Se non sono sia split1 che split2
        if split1 != seed1 or split2 != seed2:
            temp_cf = copy.copy(self.CF_list[seed1])
            temp_cf.merge(self.CF_list[seed2])
            # Se posso unirli
            if temp_cf.radius() < self.threshold:
                # Se si possono unire, trasferisce tutte le foglie del secondo nel primo
                for leaf in self.children_list[seed2].leaves():
                    self.children_list[seed1].insert_CF(leaf)
                # Svuota spazio
                if seed1 > seed2:
                    temp = seed1
                    seed1 = seed2
                    seed2 = temp
                for idx in sorted((seed1, seed2), reverse=True):
                    self.children_list.pop(idx)
                    self.CF_list.pop(idx)
                self.children_list.pop(seed2)
                self.CF_list.pop(seed2)
                # Se troppo grandi splitta
                if len(self.children_list[seed1].children_list) > self.branching_factor:
                    self.split_child(seed1)
        return self

    def paths(self, path: list[int] = []):
        # Genera tutte le possibili path fino alle foglie
        if self.is_leaf:
            yield path
        else:
            for idx, node in enumerate(self.children_list):
                copied = path + [idx]
                yield from node.paths(copied)

    def _avg_nodes_in_leaves(self) -> int:
        # Restituisce il numero medio di punti per foglia
        total_nodes = self.compute_cumulative_CF().N
        leaves_number = 0
        for _ in self.leaves():
            leaves_number += 1
        return total_nodes // leaves_number

    def leaves(self):
        # Ritorna tutti i CF delle foglie
        if self.is_leaf:
            for cf in self.CF_list:
                yield cf
        else:
            for child in self.children_list:
                yield from child.leaves()

    def __getitem__(self, path):
        current_node = self
        for node in path:
            current_node = current_node.children_list[node]
        return current_node

    def __repr__(self, level=0) -> str:
        to_return = ""
        indent = "-- " * level
        for idx, node in enumerate(self.CF_list):
            to_return += f"{indent}- {node}\n"
            if self.children_list:
                child = self.children_list[idx]
                to_return += child.__repr__(level + 1)
        return to_return

    def __sizeof__(self) -> int:
        size = object.__sizeof__(self)
        size += sum(sys.getsizeof(cf) for cf in self.CF_list)
        size += sum(child.__sizeof__() for child in self.children_list or [])
        return size
