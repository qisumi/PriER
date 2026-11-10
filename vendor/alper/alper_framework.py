#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

from data_loader import DataLoader
from encoder_topk import ANNTopKSearch
from llm_oracle import LLMOracle


class AlperFramework:
    """
    Alper Framework (Algorithm 1–3)
    """

    def __init__(
        self,
        records: Dict[int, str],
        data_loader: DataLoader,
        topk_search: ANNTopKSearch,
        llm_oracle: Optional[LLMOracle],
        budget: float,
        k: int = 15,
        max_iterations: int = 50,
        top_m: int = 5,
        alpha: float = 0.8,
        theta: float = 0.6,
        xi_llm: float = 0.95,
        L: float = 0.6,
        U: float = 0.95,
        seed: int = 42,
        input_price: float = 0.0,
        output_price: float = 0.0,
    ):
        self.records = records
        self.data_loader = data_loader
        self.topk_search = topk_search
        self.llm_oracle = llm_oracle
        self.budget = float(budget)
        self.k = k
        self.max_iterations = max_iterations
        self.top_m = top_m
        self.alpha = alpha
        self.theta = theta
        self.xi_llm = xi_llm
        self.L = L
        self.U = U
        self.seed = seed

        self.input_price = input_price
        self.output_price = output_price

        self.labels: Dict[int, int] = {}
        self.neighbors: Dict[int, List[int]] = {}
        self.weights: Dict[Tuple[int, int], float] = {}
        self.verified_edges: Set[Tuple[int, int]] = set()

        self.beta = 0.0
        self.total_cost = 0.0
        self.oracle_queries = 0

        self.window_size = 50
        self.recent_input_tokens = deque(maxlen=self.window_size)
        self.recent_output_tokens = deque(maxlen=self.window_size)

    def _build_graph(self) -> List[int]:
        """Build graph G from KNN results and initialize labels"""
        entity_ids = sorted(self.records.keys())

        topk_results = self.topk_search.get_all_topk(self.k)
        for entity_id in entity_ids:
            neighbor_ids, similarities = topk_results[entity_id]
            self.neighbors[entity_id] = neighbor_ids
            for nid, sim in zip(neighbor_ids, similarities):
                self.weights[(entity_id, nid)] = float(self.alpha * sim)

        for idx, eid in enumerate(entity_ids):
            self.labels[eid] = idx

        return entity_ids

    def _compute_label_distribution(self, r_i: int) -> Dict[int, float]:
        """Compute label distribution"""
        neighbors = self.neighbors.get(r_i, [])
        if not neighbors:
            return {}

        label_scores = defaultdict(float)
        denom = 0.0
        for r_j in neighbors:
            if r_j not in self.labels:
                continue
            w_ij = self.weights.get((r_i, r_j), 0.0)
            if w_ij <= 0:
                continue
            denom += w_ij
            c = self.labels[r_j]
            label_scores[c] += w_ij

        if denom <= 0.0:
            return {}

        return {c: v / denom for c, v in label_scores.items()}

    def _compute_graph_confidence(self, pi_i: Dict[int, float]) -> Tuple[float, float, float]:
        """Compute graph confidence"""
        if not pi_i:
            return 0.0, 0.0, 0.0

        c_star = max(pi_i, key=pi_i.get)
        xi_sem = float(pi_i[c_star])

        probs = np.array(list(pi_i.values()), dtype=float)
        probs = probs[probs > 0]
        if probs.size == 0:
            ppl = 1.0
        else:
            H = -np.sum(probs * np.log(probs))
            ppl = float(np.exp(H))
        ppl = max(1.0, ppl)

        label_count = max(len(pi_i), 2)
        xi_str = 1.0 - float(np.log(ppl) / np.log(label_count))
        xi_str = max(0.0, min(1.0, xi_str))

        xi_base = xi_sem * xi_str
        return xi_base, xi_sem, xi_str

    def _estimate_cost(self) -> float:
        """Estimate cost using sliding window of recent tokens"""
        if not self.recent_input_tokens or not self.recent_output_tokens:
            avg_in = 100.0
            avg_out = 10.0
        else:
            avg_in = float(sum(self.recent_input_tokens) / len(self.recent_input_tokens))
            avg_out = float(sum(self.recent_output_tokens) / len(self.recent_output_tokens))

        omega = avg_in * self.input_price + avg_out * self.output_price
        return max(omega, 1e-8)

    def _signal_select(self, xi_base: float) -> Tuple[bool, float]:
        """Algorithm 2: SignalSelect"""
        xi_llm = self.xi_llm
        omega = self._estimate_cost()

        d = (xi_llm - xi_base) / omega

        e = np.e
        if self.budget <= 0:
            psi = float("inf")
        else:
            psi = (self.L / e) * (self.U * e / self.L) ** (self.beta / self.budget)

        if d >= psi and self.beta + omega <= self.budget:
            return True, omega
        else:
            return False, 0.0

    def _local_update(self, r_i: int, pi_i: Dict[int, float], omega_est: float) -> bool:
        """Algorithm 3: LocalUpdate"""
        if not pi_i:
            return False

        sorted_labels = sorted(pi_i.items(), key=lambda x: x[1], reverse=True)
        top_labels = [lbl for lbl, _ in sorted_labels[: self.top_m]]

        S_i: List[int] = []
        label_to_rep: Dict[int, int] = {}

        for lbl in top_labels:
            candidates = [
                r_j
                for r_j in self.neighbors.get(r_i, [])
                if self.labels.get(r_j) == lbl
            ]
            if not candidates:
                continue

            best_rep = None
            max_sim = -1.0
            for r_j in candidates:
                sim = self.topk_search.compute_similarity(r_i, r_j)
                if sim > max_sim:
                    max_sim = sim
                    best_rep = r_j

            if best_rep is not None:
                S_i.append(best_rep)
                label_to_rep[lbl] = best_rep

        if not S_i:
            return False

        before_in = self.llm_oracle.total_input_tokens
        before_out = self.llm_oracle.total_output_tokens
        before_cost = self.llm_oracle.total_cost

        result = self.llm_oracle.query_ppl(r_i, S_i)

        after_in = self.llm_oracle.total_input_tokens
        after_out = self.llm_oracle.total_output_tokens
        after_cost = self.llm_oracle.total_cost

        delta_in = max(0, after_in - before_in)
        delta_out = max(0, after_out - before_out)
        delta_cost = max(0.0, after_cost - before_cost)

        if delta_in > 0 or delta_out > 0:
            self.recent_input_tokens.append(delta_in)
            self.recent_output_tokens.append(delta_out)

        self.oracle_queries += 1
        self.beta += delta_cost
        self.total_cost += delta_cost

        if result is None:
            for r_k in S_i:
                if r_k in self.neighbors.get(r_i, []):
                    self.neighbors[r_i].remove(r_k)
                if r_i in self.neighbors.get(r_k, []):
                    self.neighbors[r_k].remove(r_i)
            return False

        r_star, matched_records = result

        l_star = None
        for lbl, rep in label_to_rep.items():
            if rep == r_star:
                l_star = lbl
                break

        if l_star is None:
            return False

        old_label = self.labels.get(r_i)
        self.labels[r_i] = l_star

        C_l = [eid for eid, lbl in self.labels.items() if lbl == l_star]
        for r_j in C_l:
            if r_j not in self.neighbors.get(r_i, []):
                self.neighbors.setdefault(r_i, []).append(r_j)
            if r_i not in self.neighbors.get(r_j, []):
                self.neighbors.setdefault(r_j, []).append(r_i)
            self.weights[(r_i, r_j)] = 1.0
            self.weights[(r_j, r_i)] = 1.0
            self.verified_edges.add((r_i, r_j))
            self.verified_edges.add((r_j, r_i))

        for lbl, rep in label_to_rep.items():
            if lbl == l_star:
                continue
            if rep in self.neighbors.get(r_i, []):
                self.neighbors[r_i].remove(rep)
            if r_i in self.neighbors.get(rep, []):
                self.neighbors[rep].remove(r_i)

        return old_label != l_star

    def run(self) -> Tuple[List[List[int]], int, Dict]:
        """Run Alper framework main loop"""
        print("\n" + "=" * 80)
        print("Running Alper Framework")
        print("=" * 80)
        print(f"K = {self.k}, m = {self.top_m}, T_max = {self.max_iterations}")
        print(f"alpha = {self.alpha}, theta = {self.theta}")
        print(f"xi_LLM = {self.xi_llm}, L = {self.L}, U = {self.U}")
        print(f"Budget B = {self.budget:.4f} USD")
        print("=" * 80 + "\n")

        entity_ids = self._build_graph()
        print(f"Graph constructed: {len(entity_ids)} nodes\n")

        for t in range(1, self.max_iterations + 1):
            changed = False
            print(f"Iteration {t} ...")

            for r_i in entity_ids:
                if not self.neighbors.get(r_i):
                    continue

                pi_i = self._compute_label_distribution(r_i)
                if not pi_i:
                    continue

                xi_base, _, _ = self._compute_graph_confidence(pi_i)

                should_query, omega_est = self._signal_select(xi_base)

                if should_query and self.beta < self.budget:
                    updated = self._local_update(r_i, pi_i, omega_est)
                    if updated:
                        changed = True
                else:
                    c_star = max(pi_i, key=pi_i.get)
                    if pi_i[c_star] > self.theta and self.labels.get(r_i) != c_star:
                        self.labels[r_i] = c_star
                        changed = True

            num_clusters = len(set(self.labels.values()))
            budget_util = (self.beta / self.budget * 100.0) if self.budget > 0 else 0.0
            print(
                f"  Iter {t}: changed={changed}, "
                f"queries={self.oracle_queries}, "
                f"cost={self.total_cost:.4f} "
                f"({self.beta:.4f}/{self.budget:.4f}, {budget_util:.1f}%), "
                f"clusters={num_clusters}"
            )

            if not changed:
                print(f"\nConverged at iteration {t}")
                break
            if self.beta >= self.budget:
                print(f"\nBudget exhausted at iteration {t}, continue with pure graph propagation only.")

        clusters_dict = defaultdict(list)
        for eid, lbl in self.labels.items():
            clusters_dict[lbl].append(eid)
        clusters = [sorted(c) for c in clusters_dict.values()]

        budget_util = (self.beta / self.budget * 100.0) if self.budget > 0 else 0.0
        stats = {
            "total_queries": self.oracle_queries,
            "total_cost": self.total_cost,
            "budget_utilization": budget_util,
            "num_clusters": len(clusters),
            "verified_edges": len(self.verified_edges),
        }

        print("\n" + "=" * 80)
        print("Alper finished.")
        print(f"Total LLM queries: {self.oracle_queries}")
        print(f"Total cost: {self.total_cost:.4f} USD (Budget: {self.budget:.4f}, {budget_util:.1f}%)")
        print(f"Number of clusters: {len(clusters)}")
        print("=" * 80 + "\n")

        return clusters, self.oracle_queries, stats


