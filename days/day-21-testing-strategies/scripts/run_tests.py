#!/usr/bin/env python3
"""
Day 21: Testing Strategies - Test Runner Script
Comprehensive test execution with reporting and monitoring
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse

class TestRunner:
    """Comprehensive test runner with reporting and monitoring"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.results = {}
        self.start_time = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load test configuration"""
        default_config = {
            'coverage_threshold': 85,
            'performance_threshold': 5.0,
            'memory_limit_mb': 2048,
            'parallel_workers': 4,
            'output_formats': ['json', 'html', 'junit'],
            'test_suites': {
                'unit': 'tests/unit/',
                'integration': 'tests/integration/',
                'e2e': 'tests/e2e/',
                'performance': 'tests/performance/'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def run_test_suite(self, suite_name: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """Run a specific test suite"""
        
        if suite_name not in self.config['test_suites']:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        test_path = self.config['test_suites'][suite_name]
        
        print(f"🧪 Running {suite_name} tests from {test_path}")
        
        # Build pytest command
        cmd = [
            'python', '-m', 'pytest',
            test_path,
            '-v',
            '--tb=short',
            f'--cov=.',
            f'--cov-fail-under={self.config["coverage_threshold"]}',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=xml',
            f'--junit-xml=test_results/{suite_name}_results.xml',
            f'--json-report=test_results/{suite_name}_report.json'
        ]
        
        # Add parallel execution for unit tests
        if suite_name == 'unit':
            cmd.extend(['-n', str(self.config['parallel_workers'])])
        
        # Add performance-specific options
        if suite_name == 'performance':
            cmd.extend(['--benchmark-only', '--benchmark-json=test_results/benchmark.json'])
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        # Run tests
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Parse results
            suite_results = {
                'suite': suite_name,
                'status': 'passed' if result.returncode == 0 else 'failed',
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
            # Try to parse JSON report if available
            json_report_path = f'test_results/{suite_name}_report.json'
            if os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                        suite_results['detailed_results'] = json_report
                except Exception as e:
                    print(f"⚠️  Could not parse JSON report: {e}")
            
            return suite_results
            
        except subprocess.TimeoutExpired:
            return {
                'suite': suite_name,
                'status': 'timeout',
                'execution_time': 1800,
                'error': 'Test suite timed out after 30 minutes'
            }
        except Exception as e:
            return {
                'suite': suite_name,
                'status': 'error',
                'error': str(e)
            }
    
    def run_all_tests(self, suites: List[str] = None) -> Dict[str, Any]:
        """Run all or specified test suites"""
        
        if suites is None:
            suites = list(self.config['test_suites'].keys())
        
        self.start_time = time.time()
        
        print(f"🚀 Starting comprehensive test run: {', '.join(suites)}")
        print(f"📅 Started at: {datetime.now().isoformat()}")
        
        # Ensure output directory exists
        os.makedirs('test_results', exist_ok=True)
        
        overall_results = {
            'started_at': datetime.now().isoformat(),
            'suites_run': suites,
            'suite_results': {},
            'overall_status': 'running'
        }
        
        # Run each test suite
        for suite in suites:
            print(f"\n{'='*60}")
            suite_result = self.run_test_suite(suite)
            overall_results['suite_results'][suite] = suite_result
            
            # Print summary
            status_emoji = "✅" if suite_result['status'] == 'passed' else "❌"
            print(f"{status_emoji} {suite.upper()} tests: {suite_result['status']} "
                  f"({suite_result.get('execution_time', 0):.2f}s)")
        
        # Calculate overall status
        failed_suites = [s for s, r in overall_results['suite_results'].items() 
                        if r['status'] != 'passed']
        
        if not failed_suites:
            overall_results['overall_status'] = 'passed'
        else:
            overall_results['overall_status'] = 'failed'
            overall_results['failed_suites'] = failed_suites
        
        overall_results['completed_at'] = datetime.now().isoformat()
        overall_results['total_execution_time'] = time.time() - self.start_time
        
        # Save overall results
        with open('test_results/overall_results.json', 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        # Print final summary
        self._print_summary(overall_results)
        
        return overall_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test execution summary"""
        
        print(f"\n{'='*60}")
        print("📊 TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"🕐 Total execution time: {results['total_execution_time']:.2f} seconds")
        print(f"📈 Overall status: {results['overall_status'].upper()}")
        
        print(f"\n📋 Suite Results:")
        for suite, result in results['suite_results'].items():
            status_emoji = "✅" if result['status'] == 'passed' else "❌"
            print(f"  {status_emoji} {suite.upper()}: {result['status']} "
                  f"({result.get('execution_time', 0):.2f}s)")
        
        if results['overall_status'] == 'failed':
            print(f"\n❌ Failed suites: {', '.join(results.get('failed_suites', []))}")
        
        print(f"\n📁 Results saved to: test_results/")
        print(f"📊 Coverage report: htmlcov/index.html")
        
        if results['overall_status'] == 'passed':
            print(f"\n🎉 All tests passed! Ready for deployment.")
        else:
            print(f"\n🔧 Some tests failed. Please review and fix issues.")
    
    def generate_report(self, output_format: str = 'html'):
        """Generate comprehensive test report"""
        
        if not os.path.exists('test_results/overall_results.json'):
            print("❌ No test results found. Run tests first.")
            return
        
        with open('test_results/overall_results.json', 'r') as f:
            results = json.load(f)
        
        if output_format == 'html':
            self._generate_html_report(results)
        elif output_format == 'json':
            print("📄 JSON report already available at: test_results/overall_results.json")
        else:
            print(f"❌ Unsupported report format: {output_format}")
    
    def _generate_html_report(self, results: Dict[str, Any]):
        """Generate HTML test report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Results - Day 21 Testing Strategies</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .passed {{ border-left: 5px solid #4CAF50; }}
                .failed {{ border-left: 5px solid #f44336; }}
                .metrics {{ display: flex; gap: 20px; margin: 10px 0; }}
                .metric {{ background: #f9f9f9; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Day 21: Testing Strategies - Test Results</h1>
                <p><strong>Execution Time:</strong> {results['total_execution_time']:.2f} seconds</p>
                <p><strong>Overall Status:</strong> {results['overall_status'].upper()}</p>
                <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>
            </div>
        """
        
        for suite, result in results['suite_results'].items():
            status_class = 'passed' if result['status'] == 'passed' else 'failed'
            html_content += f"""
            <div class="suite {status_class}">
                <h2>{suite.upper()} Tests</h2>
                <div class="metrics">
                    <div class="metric">
                        <strong>Status:</strong> {result['status']}
                    </div>
                    <div class="metric">
                        <strong>Execution Time:</strong> {result.get('execution_time', 0):.2f}s
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open('test_results/report.html', 'w') as f:
            f.write(html_content)
        
        print("📄 HTML report generated: test_results/report.html")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Day 21 Testing Strategies - Test Runner')
    parser.add_argument('--suite', choices=['unit', 'integration', 'e2e', 'performance'], 
                       help='Run specific test suite')
    parser.add_argument('--all', action='store_true', help='Run all test suites')
    parser.add_argument('--report', choices=['html', 'json'], help='Generate test report')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--coverage-threshold', type=int, default=85, 
                       help='Coverage threshold percentage')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(args.config)
    
    if args.coverage_threshold:
        runner.config['coverage_threshold'] = args.coverage_threshold
    
    try:
        if args.report:
            runner.generate_report(args.report)
        elif args.suite:
            result = runner.run_test_suite(args.suite)
            sys.exit(0 if result['status'] == 'passed' else 1)
        elif args.all:
            results = runner.run_all_tests()
            sys.exit(0 if results['overall_status'] == 'passed' else 1)
        else:
            # Default: run all tests
            results = runner.run_all_tests()
            sys.exit(0 if results['overall_status'] == 'passed' else 1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()