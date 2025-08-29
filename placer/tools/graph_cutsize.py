from typing import List, Tuple, Dict
from collections import defaultdict
import os
import torch


class GraphCutsize:
    """
    calculate the cutsize of a hypergraph using clique graph
    """

    def __init__(self, hgr_path: str, part_path: str):
        self.hgr_path = hgr_path
        self.part_path = part_path
        self.num_vertices, self.hyperedges = self.read_hgr(hgr_path)
        self.parts = self.read_partition(part_path, self.num_vertices)

    def maximize_clique_cutsize_greedy(self) -> Tuple[List[int], int, int]:
        """
        greedy maximize clique cut under the constraint of fixed cutnet.
        return: (new partition, new clique_cut, cutnet)
        """
        n = len(self.parts)
        m = len(self.hyperedges)

        # build node -> adjacent hyperedge index
        node2hedges = [[] for _ in range(n)]
        for ei, he in enumerate(self.hyperedges):
            for v in he:
                node2hedges[v].append(ei)

        # maintain the current count of each hyperedge on both sides s0, s1
        s0 = [0] * m
        s1 = [0] * m
        for ei, he in enumerate(self.hyperedges):
            cnt0 = sum(1 for v in he if self.parts[v] == 0)
            s0[ei] = cnt0
            s1[ei] = len(he) - cnt0

        def clique_cut_current():
            total = 0
            for ei, he in enumerate(self.hyperedges):
                total += s0[ei] * s1[ei]
            return total

        def cutnet_current():
            return sum(1 for ei in range(m) if s0[ei] > 0 and s1[ei] > 0)

        def node_delta(v: int):
            """return the delta of clique cut and cutnet after flipping v"""
            p = self.parts[v]
            d_clq = 0
            d_cut = 0
            for ei in node2hedges[v]:
                a, b = s0[ei], s1[ei]  # current count
                before = (a > 0 and b > 0)
                if p == 0:
                    # v from 0 side to 1 side
                    d_clq += (a - b - 1)
                    a2, b2 = a - 1, b + 1
                else:
                    # v from 1 side to 0 side
                    d_clq += (b - a - 1)
                    a2, b2 = a + 1, b - 1
                after = (a2 > 0 and b2 > 0)
                d_cut += int(after) - int(before)
            return d_clq, d_cut

        improved = True
        while improved:
            improved = False
            best_v = -1
            best_gain = 0

            # evaluate each node (simple and reliable; if faster, only recompute the affected neighborhood)
            for v in range(n):
                d_clq, d_cut = node_delta(v)
                if d_cut == 0 and d_clq > best_gain:
                    best_gain = d_clq
                    best_v = v

            if best_v != -1:
                # accept the best feasible flip
                p = self.parts[best_v]
                self.parts[best_v] = 1 - p
                for ei in node2hedges[best_v]:
                    if p == 0:
                        s0[ei] -= 1
                        s1[ei] += 1
                    else:
                        s0[ei] += 1
                        s1[ei] -= 1
                improved = True

        return clique_cut_current(), cutnet_current()

    def minimize_clique_cutsize_greedy(self) -> Tuple[List[int], int, int]:
        """
        greedy maximize clique cut under the constraint of fixed cutnet.
        return: (new partition, new clique_cut, cutnet)
        """
        n = len(self.parts)
        m = len(self.hyperedges)

        # build node -> adjacent hyperedge index
        node2hedges = [[] for _ in range(n)]
        for ei, he in enumerate(self.hyperedges):
            for v in he:
                node2hedges[v].append(ei)

        # maintain the current count of each hyperedge on both sides s0, s1
        s0 = [0] * m
        s1 = [0] * m
        for ei, he in enumerate(self.hyperedges):
            cnt0 = sum(1 for v in he if self.parts[v] == 0)
            s0[ei] = cnt0
            s1[ei] = len(he) - cnt0

        def clique_cut_current():
            total = 0
            for ei, he in enumerate(self.hyperedges):
                total += s0[ei] * s1[ei]
            return total

        def cutnet_current():
            return sum(1 for ei in range(m) if s0[ei] > 0 and s1[ei] > 0)

        def node_delta(v: int):
            """return the delta of clique cut and cutnet after flipping v"""
            p = self.parts[v]
            d_clq = 0
            d_cut = 0
            for ei in node2hedges[v]:
                a, b = s0[ei], s1[ei]  # current count
                before = (a > 0 and b > 0)
                if p == 0:
                    # v from 0 side to 1 side
                    d_clq += (a - b - 1)
                    a2, b2 = a - 1, b + 1
                else:
                    # v from 1 side to 0 side
                    d_clq += (b - a - 1)
                    a2, b2 = a + 1, b - 1
                after = (a2 > 0 and b2 > 0)
                d_cut += int(after) - int(before)
            return d_clq, d_cut

        improved = True
        while improved:
            improved = False
            best_v = -1
            best_gain = 0

            # evaluate each node (simple and reliable; if faster, only recompute the affected neighborhood)
            for v in range(n):
                d_clq, d_cut = node_delta(v)
                if d_cut == 0 and d_clq < best_gain:
                    best_gain = d_clq
                    best_v = v

            if best_v != -1:
                # accept the best feasible flip
                p = self.parts[best_v]
                self.parts[best_v] = 1 - p
                for ei in node2hedges[best_v]:
                    if p == 0:
                        s0[ei] -= 1
                        s1[ei] += 1
                    else:
                        s0[ei] += 1
                        s1[ei] -= 1
                improved = True

        return clique_cut_current(), cutnet_current()

    def calculate(self):

        # fast method
        cut_fast = self.clique_cutsize_fast(self.hyperedges, self.parts)

        # explicitly build clique graph
        G = self.build_clique_graph(self.hyperedges)
        cut_graph = self.cutsize_from_clique_graph(G, self.parts)
        cutnet = self.hypergraph_cutsize_cutnet(self.hyperedges, self.parts)

        return cut_fast, cutnet

    def read_hgr(self, hgr_path: str) -> Tuple[int, List[List[int]]]:
        """
        read hMetis .hgr file (default: first line is "num_hyperedges num_vertices"; nodes are numbered from 1)
        return:
        - num_vertices: number of nodes
        - hyperedges: List[List[int]], each hyperedge is a list of 0-based node indices
        """
        hyperedges: List[List[int]] = []
        with open(hgr_path, "r") as f:
            lines = [
                ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("%")
            ]

        if not lines:
            raise ValueError(f"empty file: {hgr_path}")

        header = lines[0].split()
        if len(header) < 2:
            raise ValueError(f"invalid header: {lines[0]}")
        num_hyperedges = int(header[0])
        num_vertices = int(header[1])

        # 读取后续每行的超边（1-based -> 0-based）
        for ln in lines[1:]:
            he = [int(x) - 1 for x in ln.split()]
            if he:
                hyperedges.append(he)

        # tolerance: if the number of lines in the file does not match the number of hyperedges declared in the header, use the file content
        if len(hyperedges) != num_hyperedges:
            # warning: here we continue directly
            pass

        return num_vertices, hyperedges

    def read_partition(self, part_path: str, expect_n: int) -> List[int]:
        """
        read partition file: each line is a 0/1, length should be expect_n
        """
        parts: List[int] = []
        with open(part_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts.append(int(ln))

        if len(parts) != expect_n:
            raise ValueError(
                f"partition file line number ({len(parts)}) does not match the number of nodes ({expect_n})"
            )
        for v in parts:
            if v not in (0, 1):
                raise ValueError(f"partition value must be 0/1, actual: {v}")
        return parts

    def clique_cutsize_fast(self, hyperedges: List[List[int]],
                            parts: List[int]) -> int:
        """
        fast method: for each hyperedge, count the number of nodes on each side, and add n0*n1
        equivalent to constructing clique graph and summing the weights of cross-partition edges, but without explicit construction.
        """
        total = 0
        for he in hyperedges:
            n0 = sum(1 for v in he if parts[v] == 0)
            n1 = len(he) - n0
            if n0 > 0 and n1 > 0:
                total += n0 * n1
        return total

    def build_clique_graph(
            self, hyperedges: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """
        explicitly build clique graph (undirected, weighted).
        the weight of an edge is the number of hyperedges that contain it (incremented by 1 for each occurrence).
        return dictionary: {(u,v): weight}, where u < v
        """
        edge_w = defaultdict(int)
        for he in hyperedges:
            k = len(he)
            if k < 2:
                continue
            he_sorted = sorted(he)
            for i in range(k):
                u = he_sorted[i]
                for j in range(i + 1, k):
                    v = he_sorted[j]
                    edge_w[(u, v)] += 1
        return edge_w

    def cutsize_from_clique_graph(self, edge_w: Dict[Tuple[int, int], int],
                                  parts: List[int]) -> int:
        """
        sum the weights of cross-partition edges on the already built clique graph.
        """
        total = 0
        for (u, v), w in edge_w.items():
            if parts[u] != parts[v]:
                total += w
        return total

    def hypergraph_cutsize_cutnet(self, hyperedges: List[List[int]],
                                  parts: List[int]) -> int:
        """2-way cutsize (cut-net): number of hyperedges that are cut"""
        cut = 0
        for he in hyperedges:
            has0 = has1 = False
            for v in he:
                if parts[v] == 0: has0 = True
                else: has1 = True
                if has0 and has1:
                    cut += 1
                    break
        return cut


    def save_partition_to_file(self, output_path: str) -> bool:
        """
        save partition result to .part
        Args:
            output_path: 输出文件路径，例如 "circuit.hgr.part.2"
        
        """
        try:
            # 确保输出目录存在
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 写入分区结果，每行一个分区值
            with open(output_path, 'w') as f:
                for part_value in self.parts:
                    f.write(f"{part_value}\n")

            print(f"分区结果已保存到: {output_path}")
            print(f"共保存 {len(self.parts)} 个节点的分区信息")
            return True

        except Exception as e:
            print(f"保存分区文件失败: {e}")
            return False


if __name__ == "__main__":
    hgr_path = "/Users/jeannewillis/Desktop/CODE/D2D-placer/install/run_tmp/case2_hidden/circuit.hgr"
    part_path = "/Users/jeannewillis/Desktop/CODE/D2D-placer/install/run_tmp/case2_hidden/circuit.hgr.part.2"

    c = CliqueGraphCutsize(hgr_path, part_path)
    c()
