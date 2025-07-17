'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-07-17 14:00:51
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-07-17 14:23:01
FilePath: /D2D-placer/placer/tools/DreamplaceData.py
Description: 
'''
import dreamplace.PlaceDB as PlaceDB
import dreamplace.Timer as Timer
import dreamplace.configure as configure
import dreamplace.NonLinearPlace as NonLinearPlace
import time
import logging
import numpy as np
import os

class DreamplaceData:
    def __init__(self):
        self.metrics_2d = None
        self.metrics_tier = None
        self.metrics_terminal = None
        
        self.pos_2d = None
        self.pos_tier = None
        self.pos_terminal = None
        
    def database(params):
        """
        @brief Data collection for placement.
        @param params parameters
        @param placedb placement database
        """
        assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
                "CANNOT enable GPU without CUDA compiled"

        np.random.seed(params.random_seed)
        # read database
        tt = time.time()
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        logging.info("reading database takes %.2f seconds" % (time.time() - tt))

        # Read timing constraints provided in the benchmarks into out timing analysis
        # engine and then pass the timer into the placement core.
        timer = None
        if params.timing_opt_flag:
            tt = time.time()
            timer = Timer.Timer()
            timer(params, placedb)
            # This must be done to explicitly execute the parser builders.
            # The parsers in OpenTimer are all in lazy mode.
            timer.update_timing()
            logging.info("reading timer takes %.2f seconds" % (time.time() - tt))

        return placedb, timer
      
    def place(params, placedb, timer):
        """
        @brief Top API to run the entire placement flow.
        @param params parameters
        """

        # solve placement
        tt = time.time()
        placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
        logging.info("non-linear placement initialization takes %.2f seconds" %
                    (time.time() - tt))
        metrics = placer(params, placedb)
        logging.info("non-linear placement takes %.2f seconds" %
                    (time.time() - tt))

        # write placement solution
        path = "%s/%s" % (params.result_dir, params.design_name())
        if not os.path.exists(path):
            os.system("mkdir -p %s" % (path))
        gp_out_file = os.path.join(
            path,
            "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
        placedb.write(params, gp_out_file)

        return metrics, placer.pos[0]



