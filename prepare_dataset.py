import os
import json
import glob
import logging
import argparse
import time
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import setup_logging

SOLVER_PATH = './kissat/build/kissat'
DATASET_DIR = './dataset/raw_data'

def parse_args():
    parser = argparse.ArgumentParser(description='Test program argument parser')
    
    # # Add solver parameter
    # parser.add_argument('--solver', type=str, required=True,
    #                   help='name of the solver')
    
    # Add testcases parameter, supports single case or multiple cases
    parser.add_argument('--testcases', type=str, default='/Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf', 
                      help='testcase can be single case(test1), case list[test1, test2, test3], "SAT", "UNSAT" or "all"')
    
    # Add thread number parameter
    parser.add_argument('--thread_num', type=int, default=8,
                      help='number of threads (default: 8)')
    
    # Add timeout parameter
    parser.add_argument('--timeout', type=int, default=60,
                      help='timeout for each test case (default: 60 seconds)')
    
    # Add args parameter
    parser.add_argument('--args', type=str, default='',
                      help='args for solver')
    
    # Add quiet option 
    parser.add_argument('--quiet', action='store_true', default=False, help='quiet mode')
    
    args = parser.parse_args()
    
    # Testcases 
    test_case_paths = glob.glob(os.path.join(args.testcases, '*.cnf'))
    test_cases_dict = {}
    test_cases_dict['SAT'] = []
    for test_case in test_case_paths:
        test_name = os.path.basename(test_case).split('.')[0]
        test_cases_dict['SAT'].append(test_name)
    
    # Store the paths in args
    args.test_case_paths = test_case_paths
    return args, test_cases_dict


def run_single_test(test_case_path, test_args, solver_run_cmd, timeout, result_dir, quiet):
    """Run a single test case and parse its output"""
    test_case = test_case_path.split('/')[-1].split('.')[0]
    logging.info(f"🔄 Running test case: {test_case}")
    start_time = time.time()
    
    try:
        inst_name = test_case_path.split('/')[-1].replace('.cnf', '')
        solver_status_file = os.path.join(DATASET_DIR, '{}.log'.format(inst_name))
        
        cmd = f"{solver_run_cmd} -q {test_case_path} --statuslog {solver_status_file} {test_args}"
        # print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            logging.info(f"✅ {test_case} completed in {round(time.time() - start_time, 2)}s")
        except subprocess.TimeoutExpired:
            process.kill()
            logging.warning(f"⏰ Test case {test_case} timed out after {timeout} seconds")
    except Exception as e:
        logging.error(f"❌ Error running test case {test_case}: {e}")
        raise e


def test(test_case_paths, test_args, thread_num, timeout, test_cases_dict, result_dir, quiet):
    """Run test cases in parallel and collect results"""
    # 设置日志
    logger = setup_logging(result_dir)
    
    try:
        logging.info("\n" + "="*60)
        logging.info(f"🚀 Test Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*60)
        
        logging.info(f"\n📋 Test Configuration:")
        # logging.info(f"   • Solver: {solver}")
        logging.info(f"   • Thread Count: {thread_num}")
        logging.info(f"   • Timeout: {timeout}s")
        logging.info(f"   • Total Test Cases: {len(test_case_paths)}")
        
        # Configure solver and parser
        solver = SOLVER_PATH
        if not os.path.exists(solver):
            raise FileNotFoundError(f"Solver executable not found: {solver}")
        
        results = {}
        start_time = time.time()
        
        logging.info("\n⚡ Starting Test Execution...\n")
        logging.info("-"*60)
        
        # Create thread pool and run tests in parallel
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            future_to_test = {
                executor.submit(run_single_test, test_case_path, test_args, solver, timeout, result_dir, quiet): test_case_path
                for test_case_path in test_case_paths
            }
            print()
            
            completed = 0
            for future in as_completed(future_to_test):
                completed += 1
                test_case = future_to_test[future].split('/')[-1].split('.')[0]
                try:
                    future.result()
                    progress = (completed / len(test_case_paths)) * 100
                    logging.info(f"📊 Progress: {progress:.1f}% ({completed}/{len(test_case_paths)})")
                except Exception as e:
                    raise e
        
        total_time = round(time.time() - start_time, 2)
        all_time = 0
        
        logging.info("\n" + "="*60)
        logging.info(f"🏁 Test Session Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"\n📊 Summary:")
        logging.info(f"   • Runtime: {total_time}s")
        
    finally:
        # 清理日志处理器
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def test_init(solver):
    """Initialize test environment by creating result directory"""
    # Create result directory with solver name and timestamp
    result_dir = Path('result') / f'{solver}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    # Create directories
    result_dir.mkdir(parents=True, exist_ok=True)
    single_logs_dir = result_dir / 'single_logs'
    single_logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Created result directory: {result_dir}")
    print(f"📁 Created single_logs directory: {single_logs_dir}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result_dir

if __name__ == '__main__':
    args, test_cases_dict = parse_args()
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    result_dir = test_init('ipsat')  
    results = test(args.test_case_paths, args.args, args.thread_num, args.timeout, test_cases_dict, result_dir, args.quiet)
