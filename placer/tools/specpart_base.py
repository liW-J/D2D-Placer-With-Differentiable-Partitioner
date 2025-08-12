'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-08-09 23:53:41
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-08-12 00:33:26
FilePath: /D2D-placer/placer/tools/specpart_date.py
Description: 
'''
from juliacall import Main as jl
from placer.configure import compile_configurations

import logging

logger = logging.getLogger(__name__)


class SpecPartBase:

    def __init__(self, run_tmp_dir):
        super().__init__()
        self.placer_dir = compile_configurations["CMAKE_INSTALL_PREFIX"]

        self.hg_partitioning_path = compile_configurations[
            "PLACER_THIRDPARTY_DIR"] + "/HypergraphPartitioning"
        self.specpart_path = self.hg_partitioning_path + "/SpecPart"
        self.specpart_jl_path = self.specpart_path + "/SpectralRefinement.jl"

        self.hg = f"{run_tmp_dir}/circuit.hgr"
        self.pfile = f"{run_tmp_dir}/circuit.hgr.part.2"

    def partitioning(self):
        """
            @brief 
            @param 
            """

        jl.cd(self.specpart_path)
        jl.include(self.specpart_jl_path)

        jl.SpectralRefinement.SpectralHmetisRefinement(
            hg=self.hg,
            pfile=self.pfile,
            Nparts=2,
            cycles=2,
            hyperedges_threshold=300,
            ub=2,
            nev=2,
            refine_iters=2,
            best_solns=5)

        jl.cd(self.placer_dir)
        logger.info("Partitioning completed successfully")

    def part_reader(self, tier, part_reader_op):
        return part_reader_op(tier, self.specpart_path)

    def flow(self, tier, part_reader_op):
        self.partitioning()
        return self.part_reader(tier, part_reader_op)


if __name__ == "__main__":
    SpecPartBase().partitioning("case2")
