"""
Die-to-Die Placement Net Analysis Script
Analysis the type and HPWL statistics of nets in die-to-die placement results

This script analyses three types of nets:
1. nets only in top die
2. nets only in bottom die
3. crossing nets (cross-layer nets, including terminals)
"""

import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use('Agg')

# set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class D2DNetAnalyzer:

    def __init__(self, benchmark_file: str, output_file: str):
        """
        Initialize the analyzer
        
        Args:
            benchmark_file: benchmark file path (case2_hidden.txt)
            output_file: output file path (output.txt)
        """
        self.benchmark_file = benchmark_file
        self.output_file = output_file

        # store data
        self.nets = {}  # net name -> net information
        self.instances = {}  # instance name -> position and die information
        self.terminals = {}  # terminal name -> position information
        self.net_instances = defaultdict(list)  # net name -> instance list
        self.lib_cells = {}  # Store LibCell information
        self.instance_types = {}  # instance name -> lib cell type

        # statistics results
        self.top_die_nets = set()  # nets only in top die
        self.bottom_die_nets = set()  # nets only in bottom die
        self.crossing_nets = set()  # crossing net

        # HPWL statistics
        self.hpwl_stats = {
            'top_die_only': {
                'count': 0,
                'total_hpwl': 0,
                'nets': []
            },
            'bottom_die_only': {
                'count': 0,
                'total_hpwl': 0,
                'nets': []
            },
            'crossing': {
                'count': 0,
                'total_hpwl': 0,
                'nets': []
            }
        }

        # Additional statistics for crossing nets
        self.crossing_net_details = {}  # net_name -> detailed information

    def parse_benchmark_file(self):
        """Parse the benchmark file, extract net and instance information"""
        print("Parsing the benchmark file...")

        with open(self.benchmark_file, 'r') as f:
            content = f.read()

        # Parse LibCell information
        libcell_pattern = r'LibCell (MC\d+) (\d+) (\d+) (\d+)'
        libcell_matches = re.findall(libcell_pattern, content)

        for match in libcell_matches:
            cell_name, width, height, pin_count = match
            self.lib_cells[cell_name] = {
                'width': int(width),
                'height': int(height),
                'pin_count': int(pin_count),
                'pins': {}  # Store pin offset information
            }

        print(f"Found {len(self.lib_cells)} LibCell definitions")

        # Parse Pin offset information for each LibCell
        current_cell = None
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('LibCell'):
                current_cell = line.split()[1]
            elif line.startswith('Pin') and current_cell and len(
                    line.split()) >= 4:
                parts = line.split()
                pin_name = parts[1]
                try:
                    x_offset = int(parts[2])
                    y_offset = int(parts[3])
                    if current_cell in self.lib_cells:
                        self.lib_cells[current_cell]['pins'][pin_name] = {
                            'x_offset': x_offset,
                            'y_offset': y_offset
                        }
                except (ValueError, IndexError):
                    # Skip invalid pin lines
                    continue

        print(f"Parsed pin offset information for all LibCells")

        # Parse Instance information
        instance_pattern = r'Inst (C\d+) (MC\d+)'
        instance_matches = re.findall(instance_pattern, content)

        for match in instance_matches:
            inst_name, cell_type = match
            self.instance_types[inst_name] = cell_type

        print(f"Found {len(self.instance_types)} Instance definitions")

        # Parse net information
        net_section = re.search(r'NumNets (\d+)(.*?)(?=NumInst|$)', content,
                                re.DOTALL)
        if net_section:
            net_content = net_section.group(2)
            # fix the regex, use a more precise match
            net_blocks = re.findall(r'Net (N\d+) (\d+)(.*?)(?=Net N\d+|$)',
                                    net_content, re.DOTALL)

            for net_name, pin_count, pin_content in net_blocks:
                pins = re.findall(r'Pin (C\d+/P\d+)', pin_content)
                self.nets[net_name] = {
                    'pin_count': int(pin_count),
                    'pins': pins,
                    'instances': [pin.split('/')[0] for pin in pins]
                }
                self.net_instances[net_name] = [
                    pin.split('/')[0] for pin in pins
                ]

        print(f"Parsing completed, found {len(self.nets)} nets")

    def parse_output_file(self):
        """Parse the output file, extract instance position and die information"""
        print("Parsing the output file...")

        with open(self.output_file, 'r') as f:
            lines = f.readlines()

        current_section = None
        for line in lines:
            line = line.strip()

            if line.startswith('TopDiePlacement'):
                current_section = 'top'
                continue
            elif line.startswith('BottomDiePlacement'):
                current_section = 'bottom'
                continue
            elif line.startswith('NumTerminals'):
                current_section = 'terminals'
                continue

            if current_section == 'top' or current_section == 'bottom':
                if line.startswith('Inst '):
                    parts = line.split()
                    if len(parts) >= 4:
                        inst_name = parts[1]
                        x, y = int(parts[2]), int(parts[3])

                        # Get the lib cell type for this instance
                        cell_type = self.instance_types.get(inst_name, 'MC1')
                        cell_info = self.lib_cells.get(cell_type, {
                            'width': 100,
                            'height': 252
                        })

                        self.instances[inst_name] = {
                            'x': x,
                            'y': y,
                            'die': current_section,
                            'cell_type': cell_type,
                            'width': cell_info['width'],
                            'height': cell_info['height'],
                            'center_x': x + cell_info['width'] // 2,
                            'center_y': y + cell_info['height'] // 2
                        }

            elif current_section == 'terminals':
                if line.startswith('Terminal '):
                    parts = line.split()
                    if len(parts) >= 4:
                        terminal_name = parts[1]
                        x, y = int(parts[2]), int(parts[3])
                        self.terminals[terminal_name] = {'x': x, 'y': y}

        print(
            f"Parsing completed, found {len(self.instances)} instances, {len(self.terminals)} terminals"
        )

    def classify_nets(self):
        """Classify net type"""
        print("Classifying net type...")

        for net_name, net_info in self.nets.items():
            net_instances = net_info['instances']
            die_types = set()

            # check if the net is in the top or bottom die
            for inst_name in net_instances:
                if inst_name in self.instances:
                    die_types.add(self.instances[inst_name]['die'])

            # check if the net is a crossing net (including terminals)
            if net_name in self.terminals:
                self.crossing_nets.add(net_name)
            elif len(die_types) == 1:
                if 'top' in die_types:
                    self.top_die_nets.add(net_name)
                elif 'bottom' in die_types:
                    self.bottom_die_nets.add(net_name)
            elif len(die_types) > 1:
                self.crossing_nets.add(net_name)

        print(f"Classification completed:")
        print(f"  Nets only in top die: {len(self.top_die_nets)}")
        print(f"  Nets only in bottom die: {len(self.bottom_die_nets)}")
        print(f"  Crossing nets: {len(self.crossing_nets)}")

    def calculate_hpwl(self,
                       net_name: str,
                       die: str,
                       include_terminal: bool = True) -> float:
        """Calculate the HPWL of the specified net on the specified die
        
        Args:
            net_name: name of the net
            die: die name ('top' or 'bottom')
            include_terminal: whether to include terminal coordinates in HPWL calculation
        """
        if net_name not in self.nets:
            return 0.0

        net_info = self.nets[net_name]
        x_coords = []
        y_coords = []

        # collect all pin coordinates of the net on the specified die
        for pin_info in net_info['pins']:
            inst_name, pin_name = pin_info.split('/')

            if inst_name in self.instances and self.instances[inst_name][
                    'die'] == die:
                instance = self.instances[inst_name]
                cell_type = instance['cell_type']

                # Get pin offset from LibCell definition
                if cell_type in self.lib_cells and pin_name in self.lib_cells[
                        cell_type]['pins']:
                    pin_offset = self.lib_cells[cell_type]['pins'][pin_name]

                    # Calculate pin absolute position: instance bottom-left + pin offset
                    pin_x = instance['x'] + pin_offset['x_offset']
                    pin_y = instance['y'] + pin_offset['y_offset']

                    x_coords.append(pin_x)
                    y_coords.append(pin_y)

        # if the net is a crossing net and include_terminal is True, add the terminal coordinates
        if net_name in self.terminals and net_name in self.crossing_nets and include_terminal:
            terminal = self.terminals[net_name]
            x_coords.append(terminal['x'])
            y_coords.append(terminal['y'])

        if len(x_coords) < 2:
            return 0.0

        # calculate the HPWL
        hpwl_x = max(x_coords) - min(x_coords)
        hpwl_y = max(y_coords) - min(y_coords)
        return hpwl_x + hpwl_y

    def calculate_all_hpwl(self):
        """Calculate the HPWL of all nets"""
        print("Calculating the HPWL of all nets...")

        # calculate the HPWL of nets only in top die
        for net_name in self.top_die_nets:
            hpwl = self.calculate_hpwl(net_name, 'top')
            self.hpwl_stats['top_die_only']['count'] += 1
            self.hpwl_stats['top_die_only']['total_hpwl'] += hpwl
            self.hpwl_stats['top_die_only']['nets'].append({
                'net_name': net_name,
                'hpwl': hpwl
            })

        # calculate the HPWL of nets only in bottom die
        for net_name in self.bottom_die_nets:
            hpwl = self.calculate_hpwl(net_name, 'bottom')
            self.hpwl_stats['bottom_die_only']['count'] += 1
            self.hpwl_stats['bottom_die_only']['total_hpwl'] += hpwl
            self.hpwl_stats['bottom_die_only']['nets'].append({
                'net_name': net_name,
                'hpwl': hpwl
            })

        # calculate the HPWL of crossing nets
        for net_name in self.crossing_nets:
            # Calculate HPWL with terminal (original calculation)
            hpwl_top_with_terminal = self.calculate_hpwl(net_name,
                                                         'top',
                                                         include_terminal=True)
            hpwl_bottom_with_terminal = self.calculate_hpwl(
                net_name, 'bottom', include_terminal=True)
            total_hpwl_with_terminal = hpwl_top_with_terminal + hpwl_bottom_with_terminal

            # Calculate HPWL without terminal (new calculation for analysis)
            hpwl_top_without_terminal = self.calculate_hpwl(
                net_name, 'top', include_terminal=False)
            hpwl_bottom_without_terminal = self.calculate_hpwl(
                net_name, 'bottom', include_terminal=False)
            total_hpwl_without_terminal = hpwl_top_without_terminal + hpwl_bottom_without_terminal

            self.hpwl_stats['crossing']['count'] += 1
            self.hpwl_stats['crossing'][
                'total_hpwl'] += total_hpwl_with_terminal
            self.hpwl_stats['crossing']['nets'].append({
                'net_name':
                net_name,
                'hpwl_top':
                hpwl_top_with_terminal,
                'hpwl_bottom':
                hpwl_bottom_with_terminal,
                'total_hpwl':
                total_hpwl_with_terminal,
                # Add HPWL values without terminal for analysis
                'hpwl_top_without_terminal':
                hpwl_top_without_terminal,
                'hpwl_bottom_without_terminal':
                hpwl_bottom_without_terminal,
                'total_hpwl_without_terminal':
                total_hpwl_without_terminal,
                'terminal_impact_top':
                hpwl_top_with_terminal - hpwl_top_without_terminal,
                'terminal_impact_bottom':
                hpwl_bottom_with_terminal - hpwl_bottom_without_terminal,
                'terminal_impact_total':
                total_hpwl_with_terminal - total_hpwl_without_terminal
            })

        print("HPWL calculation completed")

        # Calculate detailed degree information for crossing nets
        self.calculate_crossing_net_degrees()

    def calculate_crossing_net_degrees(self):
        """Calculate detailed degree information for crossing nets"""
        print("Calculating crossing net degree information...")

        for net_name in self.crossing_nets:
            net_info = self.nets[net_name]

            # Calculate total degree
            total_degree = len(net_info['pins'])

            # Calculate degree in each die
            top_degree = 0
            bottom_degree = 0
            top_pins = []
            bottom_pins = []

            for pin_info in net_info['pins']:
                inst_name, pin_name = pin_info.split('/')
                if inst_name in self.instances:
                    die = self.instances[inst_name]['die']
                    if die == 'top':
                        top_degree += 1
                        top_pins.append(pin_info)
                    elif die == 'bottom':
                        bottom_degree += 1
                        bottom_pins.append(pin_info)

            # Store detailed information
            self.crossing_net_details[net_name] = {
                'total_degree': total_degree,
                'top_degree': top_degree,
                'bottom_degree': bottom_degree,
                'top_pins': top_pins,
                'bottom_pins': bottom_pins,
                'terminal': net_name in self.terminals
            }

        print(
            f"Degree calculation completed for {len(self.crossing_nets)} crossing nets"
        )

    def generate_statistics(self):
        """Generate the statistics report"""
        print("\n" + "=" * 60)
        print("DIE-TO-DIE PLACEMENT NET ANALYSIS REPORT")
        print("=" * 60)

        # Overall statistics
        total_nets = len(self.nets)
        total_hpwl = (self.hpwl_stats['top_die_only']['total_hpwl'] +
                      self.hpwl_stats['bottom_die_only']['total_hpwl'] +
                      self.hpwl_stats['crossing']['total_hpwl'])

        print(f"\nOverall statistics:")
        print(f"  Total net count: {total_nets}")
        print(f"  Total HPWL: {total_hpwl:,.2f}")

        # Top Die Only Nets
        print(f"\nNets only in top die:")
        print(f"  Count: {self.hpwl_stats['top_die_only']['count']}")
        print(
            f"  Total HPWL: {self.hpwl_stats['top_die_only']['total_hpwl']:,.2f}"
        )
        if self.hpwl_stats['top_die_only']['count'] > 0:
            avg_hpwl = self.hpwl_stats['top_die_only'][
                'total_hpwl'] / self.hpwl_stats['top_die_only']['count']
            print(f"  Average HPWL: {avg_hpwl:,.2f}")

        # Bottom Die Only Nets
        print(f"\nNets only in bottom die:")
        print(f"  Count: {self.hpwl_stats['bottom_die_only']['count']}")
        print(
            f"  Total HPWL: {self.hpwl_stats['bottom_die_only']['total_hpwl']:,.2f}"
        )
        if self.hpwl_stats['bottom_die_only']['count'] > 0:
            avg_hpwl = self.hpwl_stats['bottom_die_only'][
                'total_hpwl'] / self.hpwl_stats['bottom_die_only']['count']
            print(f"  Average HPWL: {avg_hpwl:,.2f}")

        # Crossing Nets
        print(f"\nCrossing nets:")
        print(f"  Count: {self.hpwl_stats['crossing']['count']}")
        print(
            f"  Total HPWL: {self.hpwl_stats['crossing']['total_hpwl']:,.2f}")
        if self.hpwl_stats['crossing']['count'] > 0:
            avg_hpwl = self.hpwl_stats['crossing'][
                'total_hpwl'] / self.hpwl_stats['crossing']['count']
            print(f"  Average HPWL: {avg_hpwl:,.2f}")

            # 分别统计上下两层的HPWL
            hpwl_top_sum = sum(net['hpwl_top']
                               for net in self.hpwl_stats['crossing']['nets'])
            hpwl_bottom_sum = sum(
                net['hpwl_bottom']
                for net in self.hpwl_stats['crossing']['nets'])
            print(f"  Top Die layer HPWL: {hpwl_top_sum:,.2f}")
            print(f"  Bottom Die layer HPWL: {hpwl_bottom_sum:,.2f}")

            # Terminal impact analysis
            print(f"\n  Terminal Impact Analysis:")
            hpwl_without_terminal_sum = sum(
                net['total_hpwl_without_terminal']
                for net in self.hpwl_stats['crossing']['nets'])
            terminal_impact_sum = sum(
                net['terminal_impact_total']
                for net in self.hpwl_stats['crossing']['nets'])

            print(
                f"    Total HPWL without terminal: {hpwl_without_terminal_sum:,.2f}"
            )
            print(f"    Total terminal impact: {terminal_impact_sum:,.2f}")
            print(
                f"    Terminal impact percentage: {terminal_impact_sum/total_hpwl*100:.2f}%"
            )

            # Average terminal impact
            avg_terminal_impact = terminal_impact_sum / self.hpwl_stats[
                'crossing']['count']
            print(
                f"    Average terminal impact per net: {avg_terminal_impact:.2f}"
            )

            # Terminal impact distribution
            positive_impacts = [
                net['terminal_impact_total']
                for net in self.hpwl_stats['crossing']['nets']
                if net['terminal_impact_total'] > 0
            ]
            negative_impacts = [
                net['terminal_impact_total']
                for net in self.hpwl_stats['crossing']['nets']
                if net['terminal_impact_total'] < 0
            ]
            zero_impacts = [
                net['terminal_impact_total']
                for net in self.hpwl_stats['crossing']['nets']
                if net['terminal_impact_total'] == 0
            ]

            print(
                f"    Nets with positive terminal impact: {len(positive_impacts)}"
            )
            print(
                f"    Nets with negative terminal impact: {len(negative_impacts)}"
            )
            print(f"    Nets with zero terminal impact: {len(zero_impacts)}")

            if positive_impacts:
                print(f"    Max positive impact: {max(positive_impacts):.2f}")
                print(
                    f"    Average positive impact: {sum(positive_impacts)/len(positive_impacts):.2f}"
                )
            if negative_impacts:
                print(f"    Max negative impact: {min(negative_impacts):.2f}")
                print(
                    f"    Average negative impact: {sum(negative_impacts)/len(negative_impacts):.2f}"
                )

            # Display degree statistics for crossing nets
            if hasattr(self,
                       'crossing_net_details') and self.crossing_net_details:
                total_degrees = [
                    info['total_degree']
                    for info in self.crossing_net_details.values()
                ]
                top_degrees = [
                    info['top_degree']
                    for info in self.crossing_net_details.values()
                ]
                bottom_degrees = [
                    info['bottom_degree']
                    for info in self.crossing_net_details.values()
                ]

                print(f"\n  Crossing Net Degree Statistics:")
                print(
                    f"    Total degree range: {min(total_degrees)} - {max(total_degrees)}"
                )
                print(
                    f"    Top die degree range: {min(top_degrees)} - {max(top_degrees)}"
                )
                print(
                    f"    Bottom die degree range: {min(bottom_degrees)} - {max(bottom_degrees)}"
                )
                print(
                    f"    Average total degree: {sum(total_degrees)/len(total_degrees):.2f}"
                )
                print(
                    f"    Average top die degree: {sum(top_degrees)/len(top_degrees):.2f}"
                )
                print(
                    f"    Average bottom die degree: {sum(bottom_degrees)/len(bottom_degrees):.2f}"
                )

        # Percentage statistics
        print(f"\nPercentage distribution:")
        print(
            f"  Top Die Only: {self.hpwl_stats['top_die_only']['count']/total_nets*100:.1f}%"
        )
        print(
            f"  Bottom Die Only: {self.hpwl_stats['bottom_die_only']['count']/total_nets*100:.1f}%"
        )
        print(
            f"  Crossing Net: {self.hpwl_stats['crossing']['count']/total_nets*100:.1f}%"
        )

        print(f"\nHPWL distribution:")
        print(
            f"  Top Die Only: {self.hpwl_stats['top_die_only']['total_hpwl']/total_hpwl*100:.1f}%"
        )
        print(
            f"  Bottom Die Only: {self.hpwl_stats['bottom_die_only']['total_hpwl']/total_hpwl*100:.1f}%"
        )
        print(
            f"  Crossing Net: {self.hpwl_stats['crossing']['total_hpwl']/total_hpwl*100:.1f}%"
        )

    def save_detailed_results(self, output_file: str):
        """Save detailed results to JSON file"""
        results = {
            'summary': {
                'total_nets': len(self.nets),
                'top_die_only_count': len(self.top_die_nets),
                'bottom_die_only_count': len(self.bottom_die_nets),
                'crossing_nets_count': len(self.crossing_nets)
            },
            'hpwl_statistics': self.hpwl_stats,
            'net_classification': {
                'top_die_only': list(self.top_die_nets),
                'bottom_die_only': list(self.bottom_die_nets),
                'crossing_nets': list(self.crossing_nets)
            }
        }

        # Add crossing net degree details if available
        if hasattr(self, 'crossing_net_details') and self.crossing_net_details:
            results['crossing_net_details'] = self.crossing_net_details

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    def run_analysis(self, result_dir_root: str):
        """Run the complete analysis"""
        print("Starting Die-to-Die Placement Net analysis...")

        self.parse_benchmark_file()
        self.parse_output_file()
        self.classify_nets()
        self.calculate_all_hpwl()
        self.generate_statistics()

        # 保存详细结果
        self.save_detailed_results(
            f'{result_dir_root}/d2d_net_analysis_results.json')

        print("\nAnalysis completed!")

    def load_analysis_results(self, file_path: str):
        """Load analysis results"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_detailed_report(self, results):
        """Create detailed report"""
        print("=" * 80)
        print("DIE-TO-DIE PLACEMENT NET Detailed Analysis Report")
        print("=" * 80)

        summary = results['summary']
        hpwl_stats = results['hpwl_statistics']

        # basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"   Total net count: {summary['total_nets']:,}")
        print(
            f"   Nets only in Top Die: {summary['top_die_only_count']:,} ({summary['top_die_only_count']/summary['total_nets']*100:.1f}%)"
        )
        print(
            f"   Nets only in Bottom Die: {summary['bottom_die_only_count']:,} ({summary['bottom_die_only_count']/summary['total_nets']*100:.1f}%)"
        )
        print(
            f"   Crossing net: {summary['crossing_nets_count']:,} ({summary['crossing_nets_count']/summary['total_nets']*100:.1f}%)"
        )

        # HPWL统计
        total_hpwl = hpwl_stats['top_die_only']['total_hpwl'] + hpwl_stats[
            'bottom_die_only']['total_hpwl'] + hpwl_stats['crossing'][
                'total_hpwl']

        print(f"\n🔗 HPWL Statistics:")
        print(f"   Total HPWL: {total_hpwl:,}")
        print(
            f"   Total HPWL of nets only in Top Die: {hpwl_stats['top_die_only']['total_hpwl']:,} ({hpwl_stats['top_die_only']['total_hpwl']/total_hpwl*100:.1f}%)"
        )
        print(
            f"   Total HPWL of nets only in Bottom Die: {hpwl_stats['bottom_die_only']['total_hpwl']:,} ({hpwl_stats['bottom_die_only']['total_hpwl']/total_hpwl*100:.1f}%)"
        )
        print(
            f"   Total HPWL of crossing nets: {hpwl_stats['crossing']['total_hpwl']:,} ({hpwl_stats['crossing']['total_hpwl']/total_hpwl*100:.1f}%)"
        )

        # average HPWL
        print(f"\n📈 Average HPWL Analysis:")
        if hpwl_stats['top_die_only']['count'] > 0:
            avg_top = hpwl_stats['top_die_only']['total_hpwl'] / hpwl_stats[
                'top_die_only']['count']
            print(f"   Average HPWL of nets only in Top Die: {avg_top:.2f}")

        if hpwl_stats['bottom_die_only']['count'] > 0:
            avg_bottom = hpwl_stats['bottom_die_only'][
                'total_hpwl'] / hpwl_stats['bottom_die_only']['count']
            print(
                f"   Average HPWL of nets only in Bottom Die: {avg_bottom:.2f}"
            )

        if hpwl_stats['crossing']['count'] > 0:
            avg_crossing = hpwl_stats['crossing']['total_hpwl'] / hpwl_stats[
                'crossing']['count']
            print(f"   Average HPWL of crossing nets: {avg_crossing:.2f}")

            # Crossing net's HPWL analysis of top and bottom layers
            hpwl_top_sum = sum(net['hpwl_top']
                               for net in hpwl_stats['crossing']['nets'])
            hpwl_bottom_sum = sum(net['hpwl_bottom']
                                  for net in hpwl_stats['crossing']['nets'])
            print(
                f"   Total HPWL of crossing nets in Top Die layer: {hpwl_top_sum:,}"
            )
            print(
                f"   Total HPWL of crossing nets in Bottom Die layer: {hpwl_bottom_sum:,}"
            )
            print(
                f"   Average HPWL of crossing nets in Top Die layer: {hpwl_top_sum/hpwl_stats['crossing']['count']:.2f}"
            )
            print(
                f"   Average HPWL of crossing nets in Bottom Die layer: {hpwl_bottom_sum/hpwl_stats['crossing']['count']:.2f}"
            )

            # Terminal impact analysis
            print(f"\n🔌 Terminal Impact Analysis:")
            hpwl_without_terminal_sum = sum(
                net.get('total_hpwl_without_terminal', 0)
                for net in hpwl_stats['crossing']['nets'])
            terminal_impact_sum = sum(
                net.get('terminal_impact_total', 0)
                for net in hpwl_stats['crossing']['nets'])

            if hpwl_without_terminal_sum > 0:
                print(
                    f"   Total HPWL without terminal: {hpwl_without_terminal_sum:,}"
                )
                print(f"   Total terminal impact: {terminal_impact_sum:,}")
                print(
                    f"   Terminal impact percentage: {terminal_impact_sum/total_hpwl*100:.2f}%"
                )

                # Average terminal impact
                avg_terminal_impact = terminal_impact_sum / hpwl_stats[
                    'crossing']['count']
                print(
                    f"   Average terminal impact per net: {avg_terminal_impact:.2f}"
                )

                # Terminal impact distribution
                positive_impacts = [
                    net.get('terminal_impact_total', 0)
                    for net in hpwl_stats['crossing']['nets']
                    if net.get('terminal_impact_total', 0) > 0
                ]
                negative_impacts = [
                    net.get('terminal_impact_total', 0)
                    for net in hpwl_stats['crossing']['nets']
                    if net.get('terminal_impact_total', 0) < 0
                ]
                zero_impacts = [
                    net.get('terminal_impact_total', 0)
                    for net in hpwl_stats['crossing']['nets']
                    if net.get('terminal_impact_total', 0) == 0
                ]

                print(
                    f"   Nets with positive terminal impact: {len(positive_impacts)}"
                )
                print(
                    f"   Nets with negative terminal impact: {len(negative_impacts)}"
                )
                print(
                    f"   Nets with zero terminal impact: {len(zero_impacts)}")

                if positive_impacts:
                    print(
                        f"   Max positive impact: {max(positive_impacts):.2f}")
                    print(
                        f"   Average positive impact: {sum(positive_impacts)/len(positive_impacts):.2f}"
                    )
                if negative_impacts:
                    print(
                        f"   Max negative impact: {min(negative_impacts):.2f}")
                    print(
                        f"   Average negative impact: {sum(negative_impacts)/len(negative_impacts):.2f}"
                    )
            else:
                print(
                    f"   Terminal impact data not available in this analysis result"
                )

        # HPWL Distribution Analysis
        print(f"\n📊 HPWL Distribution Analysis:")
        top_hpwl_list = [
            net['hpwl'] for net in hpwl_stats['top_die_only']['nets']
        ]
        bottom_hpwl_list = [
            net['hpwl'] for net in hpwl_stats['bottom_die_only']['nets']
        ]
        crossing_hpwl_list = [
            net['total_hpwl'] for net in hpwl_stats['crossing']['nets']
        ]

        if top_hpwl_list:
            print(
                f"   HPWL range of nets only in Top Die: {min(top_hpwl_list):,} - {max(top_hpwl_list):,}"
            )
            print(
                f"   Median HPWL of nets only in Top Die: {np.median(top_hpwl_list):.2f}"
            )

        if bottom_hpwl_list:
            print(
                f"   HPWL range of nets only in Bottom Die: {min(bottom_hpwl_list):,} - {max(bottom_hpwl_list):,}"
            )
            print(
                f"   Median HPWL of nets only in Bottom Die: {np.median(bottom_hpwl_list):.2f}"
            )

        if crossing_hpwl_list:
            print(
                f"   HPWL range of crossing nets: {min(crossing_hpwl_list):,} - {max(crossing_hpwl_list):,}"
            )
            print(
                f"   Median HPWL of crossing nets: {np.median(crossing_hpwl_list):.2f}"
            )

    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        # Create subplots with 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('D2D Placement Net Analysis Visualizations',
                     fontsize=16,
                     fontweight='bold')

        # Define colors for different net types
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ['Top Die Only', 'Bottom Die Only', 'Crossing Nets']

        summary = results['summary']
        hpwl_stats = results['hpwl_statistics']

        # 1. Net count distribution pie chart
        ax1 = axes[0, 0]
        sizes = [
            summary['top_die_only_count'], summary['bottom_die_only_count'],
            summary['crossing_nets_count']
        ]

        ax1.pie(sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Net count distribution')

        # 2. HPWL distribution pie chart
        ax2 = axes[0, 1]
        total_hpwl = hpwl_stats['top_die_only']['total_hpwl'] + hpwl_stats[
            'bottom_die_only']['total_hpwl'] + hpwl_stats['crossing'][
                'total_hpwl']
        hpwl_sizes = [
            hpwl_stats['top_die_only']['total_hpwl'],
            hpwl_stats['bottom_die_only']['total_hpwl'],
            hpwl_stats['crossing']['total_hpwl']
        ]

        ax2.pie(hpwl_sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90)
        ax2.set_title('HPWL distribution')

        # 3. HPWL distribution box plot with average HPWL overlay
        ax3 = axes[1, 0]
        top_hpwl = [net['hpwl'] for net in hpwl_stats['top_die_only']['nets']]
        bottom_hpwl = [
            net['hpwl'] for net in hpwl_stats['bottom_die_only']['nets']
        ]
        crossing_hpwl = [
            net['total_hpwl'] for net in hpwl_stats['crossing']['nets']
        ]

        data_to_plot = [top_hpwl, bottom_hpwl, crossing_hpwl]
        bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # set colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_title('HPWL Distribution Box Plot with Average HPWL')
        ax3.set_ylabel('HPWL')
        ax3.tick_params(axis='x', rotation=45)

        # Add average HPWL as horizontal lines on the box plot
        avg_hpwls = []

        if top_hpwl:
            avg_top = np.mean(top_hpwl)
            avg_hpwls.append(avg_top)
            # Position the line at the center of the first box (index 0)
            ax3.axhline(y=avg_top,
                        color='red',
                        linestyle='--',
                        alpha=0.8,
                        xmin=0.08,
                        xmax=0.25,
                        linewidth=2,
                        label=f'Avg: {avg_top:.0f}')
        else:
            avg_hpwls.append(0)

        if bottom_hpwl:
            avg_bottom = np.mean(bottom_hpwl)
            avg_hpwls.append(avg_bottom)
            # Position the line at the center of the second box (index 1)
            ax3.axhline(y=avg_bottom,
                        color='red',
                        linestyle='--',
                        alpha=0.8,
                        xmin=0.42,
                        xmax=0.59,
                        linewidth=2,
                        label=f'Avg: {avg_bottom:.0f}')
        else:
            avg_hpwls.append(0)

        if crossing_hpwl:
            avg_crossing = np.mean(crossing_hpwl)
            avg_hpwls.append(avg_crossing)
            # Position the line at the center of the third box (index 2)
            ax3.axhline(y=avg_crossing,
                        color='red',
                        linestyle='--',
                        alpha=0.8,
                        xmin=0.76,
                        xmax=0.93,
                        linewidth=2,
                        label=f'Avg: {avg_crossing:.0f}')
        else:
            avg_hpwls.append(0)

        # Add legend for average lines
        ax3.legend(loc='upper right', fontsize=9)

        # 4. Terminal Impact Comparison (if data available)
        ax4 = axes[1, 1]

        # Check if terminal impact data is available
        if any('total_hpwl_without_terminal' in net
               for net in hpwl_stats['crossing']['nets']):
            # Show terminal impact comparison
            crossing_with_terminal = [
                net['total_hpwl'] for net in hpwl_stats['crossing']['nets']
            ]
            crossing_without_terminal = [
                net['total_hpwl_without_terminal']
                for net in hpwl_stats['crossing']['nets']
            ]

            # Create comparison box plot
            terminal_comparison_data = [
                crossing_with_terminal, crossing_without_terminal
            ]
            terminal_labels = ['With Terminal', 'Without Terminal']
            terminal_colors = ['#ff7f0e', '#2ca02c']

            bp_terminal = ax4.boxplot(terminal_comparison_data,
                                      labels=terminal_labels,
                                      patch_artist=True)

            # Set colors
            for patch, color in zip(bp_terminal['boxes'], terminal_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax4.set_title('Crossing Net HPWL: Terminal Impact')
            ax4.set_ylabel('HPWL')
            ax4.tick_params(axis='x', rotation=45)

            # Add statistics text
            avg_with_terminal = np.mean(crossing_with_terminal)
            avg_without_terminal = np.mean(crossing_without_terminal)
            impact_percentage = (avg_with_terminal - avg_without_terminal
                                 ) / avg_without_terminal * 100

            ax4.text(
                0.02,
                0.98,
                f'Avg with terminal: {avg_with_terminal:.0f}\nAvg without terminal: {avg_without_terminal:.0f}\nImpact: {impact_percentage:+.1f}%',
                transform=ax4.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            # Show average HPWL bar chart if terminal data not available
            avg_hpwls = []
            if top_hpwl:
                avg_hpwls.append(np.mean(top_hpwl))
            else:
                avg_hpwls.append(0)

            if bottom_hpwl:
                avg_hpwls.append(np.mean(bottom_hpwl))
            else:
                avg_hpwls.append(0)

            if crossing_hpwl:
                avg_hpwls.append(np.mean(crossing_hpwl))
            else:
                avg_hpwls.append(0)

            bars = ax4.bar(labels, avg_hpwls, color=colors, alpha=0.7)
            ax4.set_title('Average HPWL of different types of nets')
            ax4.set_ylabel('Average HPWL')
            ax4.tick_params(axis='x', rotation=45)

            # add value labels on the bar chart
            for bar, value in zip(bars, avg_hpwls):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2.,
                         height + height * 0.01,
                         f'{value:.0f}',
                         ha='center',
                         va='bottom')

        plt.tight_layout()
        plt.savefig('d2d_net_analysis_visualization.png',
                    dpi=300,
                    bbox_inches='tight')
        print(
            f"   Visualizations saved to: d2d_net_analysis_visualization.png")

        # create detailed analysis chart for crossing nets
        if hpwl_stats['crossing']['nets']:
            self.create_crossing_net_analysis(hpwl_stats['crossing']['nets'],
                                              results)

    def create_crossing_net_analysis(self, crossing_nets, results):
        """create detailed analysis chart for crossing nets"""
        # 使用GridSpec创建自定义布局：上方3个图表，下方2个图表平均占满宽度
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(24, 12))
        fig.suptitle('Crossing Net detailed analysis',
                     fontsize=16,
                     fontweight='bold')

        # 创建GridSpec：2行，3列
        gs = GridSpec(2, 3, figure=fig)

        # 上方3个图表：每个占1列
        ax1 = fig.add_subplot(gs[0, 0])  # 第一行第一列
        ax2 = fig.add_subplot(gs[0, 1])  # 第一行第二列
        ax3 = fig.add_subplot(gs[0, 2])  # 第一行第三列

        # 下方2个图表：第一个占1.5列，第二个占1.5列
        ax4 = fig.add_subplot(gs[1, :1])  # 第二行，跨越前两列
        ax5 = fig.add_subplot(gs[1, 1:])  # 第二行，跨越最后一列

        # extract data
        hpwl_top = [net['hpwl_top'] for net in crossing_nets]
        hpwl_bottom = [net['hpwl_bottom'] for net in crossing_nets]
        total_hpwl = [net['total_hpwl'] for net in crossing_nets]

        # 1. Top vs Bottom HPWL scatter plot by degree (replacing the simple scatter plot)
        # Load degree information from the main results file
        try:
            # with open('d2d_net_analysis_results.json', 'r', encoding='utf-8') as f:
            full_results = results

            if 'crossing_net_details' in full_results:
                # Collect data for each degree
                degree_analysis = {}
                for net_name, net_info in full_results[
                        'crossing_net_details'].items():
                    degree = net_info['total_degree']
                    if degree not in degree_analysis:
                        degree_analysis[degree] = {
                            'top_hpwls': [],
                            'bottom_hpwls': []
                        }

                    # Find corresponding HPWL data
                    for net in crossing_nets:
                        if net['net_name'] == net_name:
                            degree_analysis[degree]['top_hpwls'].append(
                                net['hpwl_top'])
                            degree_analysis[degree]['bottom_hpwls'].append(
                                net['hpwl_bottom'])
                            break

                # Create degree-based scatter plot with rolling legend
                sorted_degrees = sorted(degree_analysis.keys())

                # 滚动图例：优先显示最大的20个degree
                if len(sorted_degrees) > 20:
                    # 选择最大的20个degree
                    visible_degrees = sorted_degrees[-20:]
                    hidden_degrees = sorted_degrees[:-20]

                    # 绘制可见的degree
                    for degree in visible_degrees:
                        top_data = degree_analysis[degree]['top_hpwls']
                        bottom_data = degree_analysis[degree]['bottom_hpwls']

                        if top_data and bottom_data:
                            min_len = min(len(top_data), len(bottom_data))
                            ax1.scatter(top_data[:min_len],
                                        bottom_data[:min_len],
                                        alpha=0.6,
                                        s=50,
                                        label=f'Degree {degree}')

                    # 绘制隐藏的degree（不显示在图例中）
                    for degree in hidden_degrees:
                        top_data = degree_analysis[degree]['top_hpwls']
                        bottom_data = degree_analysis[degree]['bottom_hpwls']

                        if top_data and bottom_data:
                            min_len = min(len(top_data), len(bottom_data))
                            ax1.scatter(
                                top_data[:min_len],
                                bottom_data[:min_len],
                                alpha=0.3,  # 降低透明度
                                s=30,  # 减小点的大小
                                color='gray',  # 使用灰色
                                label='_nolegend_')  # 不显示在图例中

                    # 添加省略号说明
                    ax1.scatter(
                        [], [],
                        alpha=0,
                        label=f'... and {len(hidden_degrees)} more degrees')

                else:
                    # 正常显示所有degree
                    for degree in sorted_degrees:
                        top_data = degree_analysis[degree]['top_hpwls']
                        bottom_data = degree_analysis[degree]['bottom_hpwls']

                        if top_data and bottom_data:
                            min_len = min(len(top_data), len(bottom_data))
                            ax1.scatter(top_data[:min_len],
                                        bottom_data[:min_len],
                                        alpha=0.6,
                                        s=50,
                                        label=f'Degree {degree}')

                # Add diagonal line
                all_top = [
                    item for sublist in
                    [degree_analysis[d]['top_hpwls'] for d in sorted_degrees]
                    for item in sublist
                ]
                all_bottom = [
                    item for sublist in [
                        degree_analysis[d]['bottom_hpwls']
                        for d in sorted_degrees
                    ] for item in sublist
                ]

                if all_top and all_bottom:
                    max_val = max(max(all_top), max(all_bottom))
                    ax1.plot([0, max_val], [0, max_val],
                             'k--',
                             alpha=0.7,
                             label='Equal HPWL')

                ax1.set_xlabel('Top Die HPWL')
                ax1.set_ylabel('Bottom Die HPWL')
                ax1.set_title('Top vs Bottom HPWL by Degree')
                ax1.grid(True, alpha=0.3)

                # 根据degree数量调整图例布局
                if len(sorted_degrees) > 20:
                    ax1.legend(bbox_to_anchor=(1.05, 1),
                               loc='upper left',
                               fontsize=7,
                               ncol=2)
                else:
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Fallback to simple scatter plot if degree info not available
                ax1.scatter(hpwl_top, hpwl_bottom, alpha=0.6, color='blue')
                ax1.plot([0, max(max(hpwl_top), max(hpwl_bottom))],
                         [0, max(max(hpwl_top), max(hpwl_bottom))],
                         'r--',
                         alpha=0.7)
                ax1.set_xlabel('Top Die HPWL')
                ax1.set_ylabel('Bottom Die HPWL')
                ax1.set_title('Top Die vs Bottom Die HPWL')
                ax1.grid(True, alpha=0.3)
        except Exception as e:
            # Fallback to simple scatter plot if error occurs
            ax1.scatter(hpwl_top, hpwl_bottom, alpha=0.6, color='blue')
            ax1.plot([0, max(max(hpwl_top), max(hpwl_bottom))],
                     [0, max(max(hpwl_top), max(hpwl_bottom))],
                     'r--',
                     alpha=0.7)
            ax1.set_xlabel('Top Die HPWL')
            ax1.set_ylabel('Bottom Die HPWL')
            ax1.set_title('Top Die vs Bottom Die HPWL')
            ax1.grid(True, alpha=0.3)

        # 2. Degree vs HPWL scatter plot with rolling legend
        # Load degree information from the main results file
        try:
            # with open('d2d_net_analysis_results.json', 'r', encoding='utf-8') as f:
            full_results = results

            if 'crossing_net_details' in full_results:
                degree_data = {}
                for net_name, net_info in full_results[
                        'crossing_net_details'].items():
                    degree = net_info['total_degree']
                    # Find corresponding HPWL data
                    for net in crossing_nets:
                        if net['net_name'] == net_name:
                            if degree not in degree_data:
                                degree_data[degree] = []
                            degree_data[degree].append(net['total_hpwl'])
                            break

                # Create degree vs HPWL scatter plot with rolling legend
                sorted_degrees = sorted(degree_data.keys())

                if len(sorted_degrees) > 20:
                    # 选择最大的20个degree
                    visible_degrees = sorted_degrees[-20:]
                    hidden_degrees = sorted_degrees[:-20]

                    # 绘制可见的degree
                    for degree in visible_degrees:
                        hpwls = degree_data[degree]
                        ax2.scatter([degree] * len(hpwls),
                                    hpwls,
                                    alpha=0.6,
                                    s=50,
                                    label=f'Degree {degree}')

                    # 绘制隐藏的degree（不显示在图例中）
                    for degree in hidden_degrees:
                        hpwls = degree_data[degree]
                        ax2.scatter([degree] * len(hpwls),
                                    hpwls,
                                    alpha=0.3,
                                    s=30,
                                    color='gray',
                                    label='_nolegend_')

                    # 添加省略号说明
                    ax2.scatter(
                        [], [],
                        alpha=0,
                        label=f'... and {len(hidden_degrees)} more degrees')

                else:
                    # 正常显示所有degree
                    for degree in sorted_degrees:
                        hpwls = degree_data[degree]
                        ax2.scatter([degree] * len(hpwls),
                                    hpwls,
                                    alpha=0.6,
                                    s=50,
                                    label=f'Degree {degree}')

                ax2.set_xlabel('Net Degree')
                ax2.set_ylabel('Total HPWL')
                ax2.set_title('Net Degree vs HPWL')
                ax2.grid(True, alpha=0.3)

                # 根据degree数量调整图例布局
                if len(sorted_degrees) > 20:
                    ax2.legend(bbox_to_anchor=(1.05, 1),
                               loc='upper left',
                               fontsize=7,
                               ncol=2)
                else:
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax2.text(0.5,
                         0.5,
                         'Degree information not available',
                         ha='center',
                         va='center',
                         transform=ax2.transAxes)
                ax2.set_title('Net Degree vs HPWL')
        except Exception as e:
            ax2.text(0.5,
                     0.5,
                     f'Error loading degree data: {str(e)}',
                     ha='center',
                     va='center',
                     transform=ax2.transAxes)
            ax2.set_title('Net Degree vs HPWL')

        # 3. NEW: Top vs Bottom degree by total Degree scatter plot
        try:
            if 'crossing_net_details' in full_results:
                # Collect degree data for each net
                degree_data = {}
                for net_name, net_info in full_results[
                        'crossing_net_details'].items():
                    total_degree = net_info['total_degree']
                    top_degree = net_info.get('top_degree', 0)
                    bottom_degree = net_info.get('bottom_degree', 0)

                    if total_degree not in degree_data:
                        degree_data[total_degree] = {
                            'top_degrees': [],
                            'bottom_degrees': []
                        }

                    degree_data[total_degree]['top_degrees'].append(top_degree)
                    degree_data[total_degree]['bottom_degrees'].append(
                        bottom_degree)

                # Create scatter plot
                sorted_total_degrees = sorted(degree_data.keys())

                # 滚动图例：优先显示最大的20个total degree
                if len(sorted_total_degrees) > 20:
                    visible_total_degrees = sorted_total_degrees[-20:]
                    hidden_total_degrees = sorted_total_degrees[:-20]

                    # 绘制可见的total degree
                    for total_degree in visible_total_degrees:
                        top_degs = degree_data[total_degree]['top_degrees']
                        bottom_degs = degree_data[total_degree][
                            'bottom_degrees']

                        if top_degs and bottom_degs:
                            min_len = min(len(top_degs), len(bottom_degs))
                            ax3.scatter(top_degs[:min_len],
                                        bottom_degs[:min_len],
                                        alpha=0.6,
                                        s=50,
                                        label=f'Total Degree {total_degree}')

                    # 绘制隐藏的total degree（不显示在图例中）
                    for total_degree in hidden_total_degrees:
                        top_degs = degree_data[total_degree]['top_degrees']
                        bottom_degs = degree_data[total_degree][
                            'bottom_degrees']

                        if top_degs and bottom_degs:
                            min_len = min(len(top_degs), len(bottom_degs))
                            ax3.scatter(top_degs[:min_len],
                                        bottom_degs[:min_len],
                                        alpha=0.3,
                                        s=30,
                                        color='gray',
                                        label='_nolegend_')

                    # 添加省略号说明
                    ax3.scatter(
                        [], [],
                        alpha=0,
                        label=
                        f'... and {len(hidden_total_degrees)} more total degrees'
                    )

                else:
                    # 正常显示所有total degree
                    for total_degree in sorted_total_degrees:
                        top_degs = degree_data[total_degree]['top_degrees']
                        bottom_degs = degree_data[total_degree][
                            'bottom_degrees']

                        if top_degs and bottom_degs:
                            min_len = min(len(top_degs), len(bottom_degs))
                            ax3.scatter(top_degs[:min_len],
                                        bottom_degs[:min_len],
                                        alpha=0.6,
                                        s=50,
                                        label=f'Total Degree {total_degree}')

                # Add diagonal line
                all_top_degs = [
                    item for sublist in [
                        degree_data[d]['top_degrees']
                        for d in sorted_total_degrees
                    ] for item in sublist
                ]
                all_bottom_degs = [
                    item for sublist in [
                        degree_data[d]['bottom_degrees']
                        for d in sorted_total_degrees
                    ] for item in sublist
                ]

                if all_top_degs and all_bottom_degs:
                    max_deg = max(max(all_top_degs), max(all_bottom_degs))
                    ax3.plot([0, max_deg], [0, max_deg],
                             'k--',
                             alpha=0.7,
                             label='Equal Degree')

                ax3.set_xlabel('Top Die Degree')
                ax3.set_ylabel('Bottom Die Degree')
                ax3.set_title('Top vs Bottom Degree by Total Degree')
                ax3.grid(True, alpha=0.3)

                # 根据total degree数量调整图例布局
                if len(sorted_total_degrees) > 20:
                    ax3.legend(bbox_to_anchor=(1.05, 1),
                               loc='upper left',
                               fontsize=7,
                               ncol=2)
                else:
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            else:
                ax3.text(0.5,
                         0.5,
                         'Degree information not available',
                         ha='center',
                         va='center',
                         transform=ax3.transAxes)
                ax3.set_title('Top vs Bottom Degree by Total Degree')
        except Exception as e:
            ax3.text(0.5,
                     0.5,
                     f'Error creating degree analysis: {str(e)}',
                     ha='center',
                     va='center',
                     transform=ax3.transAxes)
            ax3.set_title('Top vs Bottom Degree by Total Degree')

        # 4. HPWL distribution histogram
        ax4.hist(total_hpwl,
                 bins=50,
                 alpha=0.7,
                 color='green',
                 edgecolor='black')
        ax4.set_xlabel('Total HPWL')
        ax4.set_ylabel('Net count')
        ax4.set_title('Crossing Net HPWL distribution')
        ax4.grid(True, alpha=0.3)

        # 5. Degree distribution and HPWL statistics with rolling legend
        try:
            if 'crossing_net_details' in full_results:
                degree_stats = {}
                for net_name, net_info in full_results[
                        'crossing_net_details'].items():
                    degree = net_info['total_degree']
                    if degree not in degree_stats:
                        degree_stats[degree] = {
                            'count': 0,
                            'total_hpwl': 0,
                            'hpwls': []
                        }

                    # Find corresponding HPWL data
                    for net in crossing_nets:
                        if net['net_name'] == net_name:
                            degree_stats[degree]['count'] += 1
                            degree_stats[degree]['total_hpwl'] += net[
                                'total_hpwl']
                            degree_stats[degree]['hpwls'].append(
                                net['total_hpwl'])
                            break

                # Create degree distribution bar chart with HPWL statistics
                degrees = sorted(degree_stats.keys())

                # 始终显示全部degree，不使用滚动图例
                counts = [degree_stats[d]['count'] for d in degrees]
                avg_hpwls = [
                    degree_stats[d]['total_hpwl'] / degree_stats[d]['count']
                    if degree_stats[d]['count'] > 0 else 0 for d in degrees
                ]

                # 创建柱状图
                bars = ax5.bar(degrees, counts, color='skyblue', alpha=0.7)
                ax5.set_xlabel('Net Degree')
                ax5.set_ylabel('Net Count', color='skyblue')
                ax5.set_title('Degree Distribution and Average HPWL')
                ax5.grid(True, alpha=0.3)

                # 添加HPWL线图
                ax5_twin = ax5.twinx()
                line = ax5_twin.plot(degrees,
                                     avg_hpwls,
                                     'r-o',
                                     linewidth=2,
                                     markersize=6,
                                     label='Average HPWL')
                ax5_twin.set_ylabel('Average HPWL', color='red')
                ax5_twin.tick_params(axis='y', labelcolor='red')

                # 添加数值标签
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width() / 2.,
                             height + 0.1,
                             f'{count}',
                             ha='center',
                             va='bottom',
                             color='blue')

                for i, (degree, avg_hpwl) in enumerate(zip(degrees,
                                                           avg_hpwls)):
                    ax5_twin.text(degree,
                                  avg_hpwl + max(avg_hpwls) * 0.02,
                                  f'{avg_hpwl:.0f}',
                                  ha='center',
                                  va='bottom',
                                  color='red')

                # 添加图例
                ax5_twin.legend(loc='upper right')
            else:
                ax5.text(0.5,
                         0.5,
                         'Degree information not available',
                         ha='center',
                         va='center',
                         transform=ax5.transAxes)
                ax5.set_title('Degree Distribution and HPWL Statistics')
        except Exception as e:
            ax5.text(0.5,
                     0.5,
                     f'Error creating degree analysis: {str(e)}',
                     ha='center',
                     va='center',
                     transform=ax5.transAxes)
            ax5.set_title('Degree Distribution and HPWL Statistics')

        # 不再需要隐藏第6个子图，因为我们已经使用GridSpec自定义了布局

        plt.tight_layout()
        plt.savefig('crossing_net_analysis.png', dpi=300, bbox_inches='tight')
        print(
            f"   Crossing net analysis chart saved to: crossing_net_analysis.png"
        )

    def analyze_terminal_impact(self, results):
        """Analyze terminal impact on crossing nets"""
        print("=" * 80)
        print("TERMINAL IMPACT ON CROSSING NETS - DETAILED ANALYSIS")
        print("=" * 80)

        hpwl_stats = results['hpwl_statistics']
        crossing_nets = hpwl_stats['crossing']['nets']

        if not crossing_nets or 'total_hpwl_without_terminal' not in crossing_nets[
                0]:
            print(
                "❌ Terminal impact data not available in this analysis result")
            return

        print(f"\n📊 Overview:")
        print(f"   Total crossing nets: {len(crossing_nets)}")

        # Calculate basic statistics
        total_with_terminal = sum(net['total_hpwl'] for net in crossing_nets)
        total_without_terminal = sum(net['total_hpwl_without_terminal']
                                     for net in crossing_nets)
        absolute_impact = total_with_terminal - total_without_terminal
        relative_impact = (absolute_impact / total_without_terminal
                           ) * 100 if total_without_terminal > 0 else 0

        print(f"   Total HPWL with terminal: {total_with_terminal:,}")
        print(f"   Total HPWL without terminal: {total_without_terminal:,}")
        print(f"   Absolute terminal impact: {absolute_impact:+,}")
        print(f"   Relative terminal impact: {relative_impact:+.2f}%")

        # Terminal impact distribution analysis
        print(f"\n🔍 Terminal Impact Distribution:")

        positive_impacts = [
            net['terminal_impact_total'] for net in crossing_nets
            if net['terminal_impact_total'] > 0
        ]
        negative_impacts = [
            net['terminal_impact_total'] for net in crossing_nets
            if net['terminal_impact_total'] < 0
        ]
        zero_impacts = [
            net['terminal_impact_total'] for net in crossing_nets
            if net['terminal_impact_total'] == 0
        ]

        print(
            f"   Nets with positive terminal impact: {len(positive_impacts)} ({len(positive_impacts)/len(crossing_nets)*100:.1f}%)"
        )
        print(
            f"   Nets with negative terminal impact: {len(negative_impacts)} ({len(negative_impacts)/len(crossing_nets)*100:.1f}%)"
        )
        print(
            f"   Nets with zero terminal impact: {len(zero_impacts)} ({len(zero_impacts)/len(crossing_nets)*100:.1f}%)"
        )

        if positive_impacts:
            print(f"   Max positive impact: {max(positive_impacts):.2f}")
            print(
                f"   Average positive impact: {np.mean(positive_impacts):.2f}")
            print(
                f"   Median positive impact: {np.median(positive_impacts):.2f}"
            )

        if negative_impacts:
            print(f"   Max negative impact: {min(negative_impacts):.2f}")
            print(
                f"   Average negative impact: {np.mean(negative_impacts):.2f}")
            print(
                f"   Median negative impact: {np.median(negative_impacts):.2f}"
            )

        # Layer-specific analysis
        print(f"\n🏗️  Layer-Specific Terminal Impact:")

        top_impacts = [net['terminal_impact_top'] for net in crossing_nets]
        bottom_impacts = [
            net['terminal_impact_bottom'] for net in crossing_nets
        ]

        print(f"   Top die terminal impact:")
        print(f"     Total: {sum(top_impacts):+,}")
        print(f"     Average: {np.mean(top_impacts):.2f}")
        print(f"     Max: {max(top_impacts):.2f}")

        print(f"   Bottom die terminal impact:")
        print(f"     Total: {sum(bottom_impacts):+,}")
        print(f"     Average: {np.mean(bottom_impacts):.2f}")
        print(f"     Max: {max(bottom_impacts):.2f}")

        # Impact magnitude analysis
        print(f"\n📈 Impact Magnitude Analysis:")

        impact_magnitudes = [
            abs(net['terminal_impact_total']) for net in crossing_nets
        ]
        impact_magnitudes.sort()

        print(f"   Impact magnitude statistics:")
        print(f"     Min: {min(impact_magnitudes):.2f}")
        print(f"     Max: {max(impact_magnitudes):.2f}")
        print(f"     Mean: {np.mean(impact_magnitudes):.2f}")
        print(f"     Median: {np.median(impact_magnitudes):.2f}")
        print(
            f"     75th percentile: {np.percentile(impact_magnitudes, 75):.2f}"
        )
        print(
            f"     90th percentile: {np.percentile(impact_magnitudes, 90):.2f}"
        )

        # Categorize by impact magnitude
        small_impact = [mag for mag in impact_magnitudes if mag <= 1000]
        medium_impact = [
            mag for mag in impact_magnitudes if 1000 < mag <= 3000
        ]
        large_impact = [mag for mag in impact_magnitudes if mag > 3000]

        print(f"   Impact magnitude distribution:")
        print(
            f"     Small impact (≤1000): {len(small_impact)} nets ({len(small_impact)/len(crossing_nets)*100:.1f}%)"
        )
        print(
            f"     Medium impact (1000-3000): {len(medium_impact)} nets ({len(medium_impact)/len(crossing_nets)*100:.1f}%)"
        )
        print(
            f"     Large impact (>3000): {len(large_impact)} nets ({len(large_impact)/len(crossing_nets)*100:.1f}%)"
        )

        return {
            'positive_impacts': positive_impacts,
            'negative_impacts': negative_impacts,
            'zero_impacts': zero_impacts,
            'top_impacts': top_impacts,
            'bottom_impacts': bottom_impacts,
            'impact_magnitudes': impact_magnitudes,
            'total_with_terminal': total_with_terminal,
            'total_without_terminal': total_without_terminal,
            'absolute_impact': absolute_impact,
            'relative_impact': relative_impact
        }

    def create_terminal_impact_visualizations(self, impact_data):
        """Create visualizations for terminal impact analysis"""
        print(f"\n📊 Creating terminal impact visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Terminal Impact on Crossing Nets - Detailed Analysis',
                     fontsize=16,
                     fontweight='bold')

        # 1. Terminal impact distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(impact_data['positive_impacts'],
                 bins=20,
                 alpha=0.7,
                 color='green',
                 label='Positive Impact')
        ax1.hist(impact_data['negative_impacts'],
                 bins=20,
                 alpha=0.7,
                 color='red',
                 label='Negative Impact')
        ax1.set_xlabel('Terminal Impact')
        ax1.set_ylabel('Number of Nets')
        ax1.set_title('Terminal Impact Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Layer-specific impact comparison
        ax2 = axes[0, 1]
        layer_data = [
            impact_data['top_impacts'], impact_data['bottom_impacts']
        ]
        layer_labels = ['Top Die', 'Bottom Die']
        bp2 = ax2.boxplot(layer_data, labels=layer_labels, patch_artist=True)

        colors = ['#ff7f0e', '#2ca02c']
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_title('Layer-Specific Terminal Impact')
        ax2.set_ylabel('Terminal Impact')
        ax2.grid(True, alpha=0.3)

        # 3. Impact magnitude cumulative distribution
        ax3 = axes[0, 2]
        sorted_magnitudes = sorted(impact_data['impact_magnitudes'])
        cumulative_prob = np.arange(
            1,
            len(sorted_magnitudes) + 1) / len(sorted_magnitudes)
        ax3.plot(sorted_magnitudes, cumulative_prob, 'b-', linewidth=2)
        ax3.set_xlabel('Impact Magnitude')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Terminal Impact Magnitude CDF')
        ax3.grid(True, alpha=0.3)

        # 4. Before vs After comparison
        ax4 = axes[1, 0]
        before_after_data = [
            impact_data['total_without_terminal'],
            impact_data['total_with_terminal']
        ]
        before_after_labels = ['Without Terminal', 'With Terminal']
        colors = ['#1f77b4', '#ff7f0e']

        bars = ax4.bar(before_after_labels,
                       before_after_data,
                       color=colors,
                       alpha=0.7)
        ax4.set_title('Total HPWL: Terminal Impact')
        ax4.set_ylabel('Total HPWL')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, before_after_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.,
                     height + height * 0.01,
                     f'{value:,.0f}',
                     ha='center',
                     va='bottom')

        # 5. Impact magnitude histogram
        ax5 = axes[1, 1]
        ax5.hist(impact_data['impact_magnitudes'],
                 bins=30,
                 alpha=0.7,
                 color='purple',
                 edgecolor='black')
        ax5.set_xlabel('Impact Magnitude')
        ax5.set_ylabel('Number of Nets')
        ax5.set_title('Terminal Impact Magnitude Distribution')
        ax5.grid(True, alpha=0.3)

        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary_text = f"""Terminal Impact Summary:
        
    Total Impact: {impact_data['absolute_impact']:+,}
    Relative Impact: {impact_data['relative_impact']:+.2f}%

    Impact Distribution:
    • Positive: {len(impact_data['positive_impacts'])} nets
    • Negative: {len(impact_data['negative_impacts'])} nets  
    • Zero: {len(impact_data['zero_impacts'])} nets

    Magnitude Statistics:
    • Mean: {np.mean(impact_data['impact_magnitudes']):.1f}
    • Median: {np.median(impact_data['impact_magnitudes']):.1f}
    • 90th percentile: {np.percentile(impact_data['impact_magnitudes'], 90):.1f}"""

        ax6.text(0.1,
                 0.9,
                 summary_text,
                 transform=ax6.transAxes,
                 fontsize=12,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('terminal_impact_analysis.png',
                    dpi=300,
                    bbox_inches='tight')
        print(
            f"   Terminal impact visualizations saved to: terminal_impact_analysis.png"
        )


def main():
    """Main function"""
    # file paths
    benchmark_file = "benchmarks/iccad2022/case2_hidden.txt"
    output_file = "install/results/case2_hidden/2025-08-26_13-47-43/output.txt"

    # create the analyzer and run the analysis
    analyzer = D2DNetAnalyzer(benchmark_file, output_file)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
