'''
Date: 2025-08-09 23:53:41
LastEditTime: 2025-10-29 23:32:40
FilePath: /D2D-placer/placer/tools/specpart_base.py
Description: 
'''
from juliacall import Main as jl
from placer.configure import compile_configurations

import logging
import os

logger = logging.getLogger(__name__)


class SpecPartBase:

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.placer_dir = compile_configurations["CMAKE_INSTALL_PREFIX"]

        self.hg_partitioning_path = compile_configurations[
            "PLACER_THIRDPARTY_DIR"] + "/HypergraphPartitioning"
        self.specpart_path = self.hg_partitioning_path + "/SpecPart"
        self.specpart_jl_path = self.specpart_path + "/SpectralRefinement.jl"

        self.hg = f"{params.run_tmp_dir_root}/{params.case_name}.hgr"
        self.pfile = f"{params.run_tmp_dir_root}/{params.case_name}.hgr.part.2"

        self.hmetis_path = f"{self.placer_dir}/bin/hmetis"

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

    def flow(self, hgr_generator_op, parts_reader_op):
        hgr_generator_op(self.params.case_name)
        # run hmetis
        cmd =  f"{self.hmetis_path} {self.hg} 2 2 10 1 1 0 1 0"
        os.system(cmd)

        self.partitioning()
        if os.path.exists(
                f"{self.specpart_path}/{self.params.case_name}.hgr.part.2"):
            os.rename(
                f"{self.specpart_path}/{self.params.case_name}.hgr.part.2",
                self.pfile)
            tier = parts_reader_op(self.params.case_name)
        else:
            tier = parts_reader_op(self.params.case_name)
            logger.info(
                "SpecPart partition result file not found, use parts_reader_op to generate partition result"
            )
        return tier


if __name__ == "__main__":
    SpecPartBase().partitioning()
