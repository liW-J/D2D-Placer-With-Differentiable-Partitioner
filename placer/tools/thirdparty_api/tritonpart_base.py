'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-09-10 15:19:57
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-09-15 22:51:16
FilePath: /D2D-placer/placer/tools/thirdparty_api/tritonpart_base.py
Description: 
'''
from placer.configure import compile_configurations
import logging
import os

logger = logging.getLogger(__name__)


class TritonPartBase:

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.tritonpart_tcl_path = compile_configurations[
            "PLACER_SOURCE_DIR"] + "/scripts/tritonpart.tcl"

        os.environ["TRITONPART_CASE_NAME"] = params.case_name
        os.environ["PLACER_RUNTMP_DIR"] = params.run_tmp_dir_root

    def convert_gp_pl_to_embedding(self, input_file: str):
        """
        convert .gp.pl to .embedding.dat
        
        Args:
            input_file: input .gp.pl file path
            output_file: output .embedding.dat file path (optional)
        """

        # check if input file exists
        if not os.path.exists(input_file):
            print(f"error: input file {input_file} does not exist")
            return False

        # determine output file path
        output_file = self.placement_file
        coordinates = []

        # read and parse .gp.pl file
        try:
            with open(input_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # skip empty lines and header lines
                    if not line or line.startswith('UCLA'):
                        continue

                    # parse format: node name x coordinate y coordinate : FS
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            coordinates.append((x, y))
                        except ValueError:
                            print(
                                f"warning: failed to parse line {line_num}: {line}"
                            )
                            continue
                    else:
                        print(
                            f"warning: line {line_num} format is incorrect: {line}"
                        )
                        continue
        except Exception as e:
            print(f"error: failed to read file: {e}")
            return False

        print(f"successfully parsed {len(coordinates)} nodes")

        if not coordinates:
            print("error: no valid coordinates found")
            return False

        # find coordinate range
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        print(f"coordinate range:")
        print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")

        # normalize coordinates to [0, 1] range
        normalized_coords = []
        for x, y in coordinates:
            norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
            norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
            normalized_coords.append((norm_x, norm_y))

        # write to output file
        try:
            with open(output_file, 'w') as f:
                for x, y in normalized_coords:
                    f.write(f"{x:.15e}, {y:.15e}\n")

            print(
                f"successfully wrote {len(normalized_coords)} normalized coordinates to {output_file}"
            )
            return True

        except Exception as e:
            print(f"error: failed to write file: {e}")
            return False

    def partitioning(self):

        exit_code = os.system(f"openroad {self.tritonpart_tcl_path}")
        if exit_code != 0:
            print(f"error: failed to run tritonpart.tcl")
            return False
        return True

    def flow(self, hgr_generator_op, parts_reader_op):
        hgr_generator_op(self.params.case_name)
        self.partitioning()
        tier = parts_reader_op(self.params.case_name)
        return tier


if __name__ == "__main__":
    TritonPartBase().partitioning("case2")
