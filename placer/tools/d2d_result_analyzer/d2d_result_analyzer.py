#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-08-31 20:29:01
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-10-18 17:03:09
FilePath: /D2D-placer/placer/tools/d2d_result_analyzer/d2d_result_analyzer.py
Description: 
    D2D Placement Result Analyzer - Unified Interface
    This module provides a unified interface for analyzing D2D placement results,
    integrating all analysis functionalities including net analysis, detailed reports,
    and terminal impact analysis.
'''

import os
import json
import logging
from typing import Dict, List, Optional
import numpy as np
# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use('Agg')

# Import analysis modules
from placer.tools.d2d_result_analyzer.d2d_net_analyzer import D2DNetAnalyzer


class D2DResultAnalyzer:
    """
    Unified interface for D2D placement result analysis
    
    This class integrates all analysis functionalities:
    1. Basic net analysis and classification
    2. Detailed HPWL statistics and reports
    3. Terminal impact analysis
    4. Comprehensive visualizations
    """

    def __init__(self,
                 benchmark_file: str,
                 result_dir: str = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the analyzer
        
        Args:
            benchmark_file: Path to benchmark file (e.g., case2_hidden.txt)
            output_file: Path to placement output file
            result_dir: Directory to save analysis results (optional)
            flattened_pl_file: Path to flattened-2d placement file (optional)
            logger: Logger instance for output (optional)
        """

        self.benchmark_file = benchmark_file
        self.result_dir = result_dir

        self.output_file = os.path.join(result_dir, "output.txt")
        self.flattened_pl_file = os.path.join(
            result_dir, "flattened-2d/flattened-2d.gp.pl")

        self.logger = logger or logging.getLogger(__name__)

        # Ensure result directory exists
        os.makedirs(self.result_dir, exist_ok=True)

        # Analysis results storage
        self.analysis_results = None
        self.net_analyzer = None

        # Output file paths
        self.results_json = os.path.join(self.result_dir,
                                         'd2d_net_analysis_results.json')
        self.main_viz = os.path.join(self.result_dir,
                                     'd2d_net_analysis_visualization.png')
        self.crossing_viz = os.path.join(self.result_dir,
                                         'crossing_net_analysis.png')
        self.flattened_2d_viz = os.path.join(
            self.result_dir, 'flattened_2d_hpwl_comparison.png')
        self.terminal_viz = os.path.join(self.result_dir,
                                         'terminal_impact_analysis.png')

        self.logger.info(f"D2D Result Analyzer initialized")
        self.logger.info(f"Result directory: {self.result_dir}")

    def run_basic_analysis(self) -> Dict:
        """
        Run basic net analysis and classification
        
        Returns:
            Dictionary containing basic analysis results
        """
        self.logger.info("Starting basic net analysis...")

        try:
            # Create and run the net analyzer
            self.net_analyzer = D2DNetAnalyzer(self.benchmark_file,
                                               self.output_file,
                                               self.flattened_pl_file)
            self.net_analyzer.run_analysis(self.result_dir)

            # Load the results
            with open(self.results_json, 'r', encoding='utf-8') as f:
                self.analysis_results = json.load(f)

            self.logger.info("Basic analysis completed successfully")
            return self.analysis_results

        except Exception as e:
            self.logger.error(f"Basic analysis failed: {str(e)}")
            raise

    def generate_detailed_report(self) -> str:
        """
        Generate detailed analysis report
        
        Returns:
            Report content as string
        """
        if not self.analysis_results:
            raise ValueError(
                "Analysis results not available. Run basic_analysis first.")

        self.logger.info("Generating detailed report...")

        # Capture the report output
        import io
        import sys

        # Redirect stdout to capture report
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        try:
            self.net_analyzer.create_detailed_report(self.analysis_results)
            report_content = new_stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Save report to file
        report_file = os.path.join(self.result_dir,
                                   'detailed_analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"Detailed report saved to: {report_file}")
        return report_content

    def create_visualizations(self) -> List[str]:
        """
        Create comprehensive visualizations
        
        Returns:
            List of created visualization file paths
        """
        if not self.analysis_results:
            raise ValueError(
                "Analysis results not available. Run basic_analysis first.")

        self.logger.info("Creating visualizations...")

        created_files = []

        try:
            # Create main visualizations
            self.net_analyzer.create_visualizations(self.analysis_results)

            # Move files to result directory if they were created elsewhere
            if os.path.exists('d2d_net_analysis_visualization.png'):
                os.rename('d2d_net_analysis_visualization.png', self.main_viz)
                created_files.append(self.main_viz)

            if os.path.exists('crossing_net_analysis.png'):
                os.rename('crossing_net_analysis.png', self.crossing_viz)
                created_files.append(self.crossing_viz)

            if os.path.exists('flattened_2d_hpwl_comparison.png'):
                os.rename('flattened_2d_hpwl_comparison.png',
                          self.flattened_2d_viz)
                created_files.append(self.flattened_2d_viz)

            self.logger.info("Main visualizations created successfully")

        except Exception as e:
            self.logger.error(f"Main visualization creation failed: {str(e)}")

        return created_files

    def analyze_terminal_impact(self) -> Optional[Dict]:
        """
        Analyze terminal impact on crossing nets
        
        Returns:
            Terminal impact analysis data or None if not available
        """
        if not self.analysis_results:
            raise ValueError(
                "Analysis results not available. Run basic_analysis first.")

        self.logger.info("Analyzing terminal impact...")

        try:
            # Check if terminal impact data is available
            crossing_nets = self.analysis_results.get(
                'hpwl_statistics', {}).get('crossing', {}).get('nets', [])
            if not crossing_nets or 'total_hpwl_without_terminal' not in crossing_nets[
                    0]:
                self.logger.warning("Terminal impact data not available")
                return None

            # Run terminal impact analysis
            impact_data = self.net_analyzer.analyze_terminal_impact(
                self.analysis_results)

            # Create terminal impact visualizations
            self.net_analyzer.create_terminal_impact_visualizations(
                impact_data)

            # Move file to result directory
            if os.path.exists('terminal_impact_analysis.png'):
                os.rename('terminal_impact_analysis.png', self.terminal_viz)

            self.logger.info("Terminal impact analysis completed successfully")
            return impact_data

        except Exception as e:
            self.logger.error(f"Terminal impact analysis failed: {str(e)}")
            return None

    def run_comprehensive_analysis(self,
                                   include_terminal_analysis: bool = True,
                                   save_intermediate: bool = False) -> Dict:
        """
        Run comprehensive analysis including all components
        
        Args:
            include_terminal_analysis: Whether to include terminal impact analysis
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Complete analysis results dictionary
        """
        self.logger.info("Starting comprehensive D2D placement analysis...")

        analysis_summary = {
            'status': 'success',
            'files_created': [],
            'analysis_results': None,
            'terminal_impact': None,
            'errors': []
        }

        try:
            # Step 1: Basic net analysis
            self.logger.info("Step 1/4: Running basic net analysis...")
            analysis_summary['analysis_results'] = self.run_basic_analysis()

            # Step 2: Generate detailed report
            self.logger.info("Step 2/4: Generating detailed report...")
            report_content = self.generate_detailed_report()
            analysis_summary['files_created'].append(
                'detailed_analysis_report.txt')

            # Step 3: Create visualizations
            self.logger.info("Step 3/4: Creating visualizations...")
            viz_files = self.create_visualizations()
            analysis_summary['files_created'].extend(
                [os.path.basename(f) for f in viz_files])

            # Step 4: Terminal impact analysis (optional)
            if include_terminal_analysis:
                self.logger.info("Step 4/4: Analyzing terminal impact...")
                terminal_data = self.analyze_terminal_impact()
                if terminal_data:
                    analysis_summary['terminal_impact'] = terminal_data
                    analysis_summary['files_created'].append(
                        'terminal_impact_analysis.png')

            self.logger.info("Comprehensive analysis completed successfully!")

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg)
            analysis_summary['status'] = 'failed'
            analysis_summary['errors'].append(error_msg)

        # Save analysis summary
        summary_file = os.path.join(self.result_dir, 'analysis_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

        analysis_summary['files_created'].append('analysis_summary.json')

        return analysis_summary

    def get_analysis_summary(self) -> Dict:
        """
        Get a summary of the analysis results
        
        Returns:
            Summary dictionary
        """
        if not self.analysis_results:
            return {'status': 'no_analysis_run'}

        summary = self.analysis_results.get('summary', {})
        hpwl_stats = self.analysis_results.get('hpwl_statistics', {})

        # Calculate total HPWL
        total_hpwl = (
            hpwl_stats.get('top_die_only', {}).get('total_hpwl', 0) +
            hpwl_stats.get('bottom_die_only', {}).get('total_hpwl', 0) +
            hpwl_stats.get('crossing', {}).get('total_hpwl', 0))

        return {
            'total_nets':
            summary.get('total_nets', 0),
            'top_die_only_count':
            summary.get('top_die_only_count', 0),
            'bottom_die_only_count':
            summary.get('bottom_die_only_count', 0),
            'crossing_nets_count':
            summary.get('crossing_nets_count', 0),
            'total_hpwl':
            total_hpwl,
            'files_created': [
                os.path.basename(self.results_json),
                os.path.basename(self.main_viz),
                os.path.basename(self.crossing_viz),
                os.path.basename(self.flattened_2d_viz),
                os.path.basename(self.terminal_viz)
            ]
        }

    def cleanup_temp_files(self):
        """Clean up temporary files created during analysis"""
        temp_files = [
            'd2d_net_analysis_visualization.png', 'crossing_net_analysis.png',
            'terminal_impact_analysis.png'
        ]

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove temporary file {temp_file}: {e}")
