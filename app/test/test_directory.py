#!/usr/bin/env python3
"""
Directory Structure Test - Validates the results directory setup

This script tests the directory validation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from colorama import Fore, init
from utils.file_validator import validate_directory_structure, get_directory_info
from config import Config

# Initialize colorama
init()

def test_directory_validation():
    """Test the directory validation functionality."""
    print(Fore.MAGENTA + "=== Directory Structure Validation Test ===\n")
    
    # Test 1: Validate base results directory
    print(Fore.CYAN + "Test 1: Validating base results directory...")
    result = validate_directory_structure(Config.RESULTS_BASE_DIR)
    print(f"Result: {'✅ PASS' if result else '❌ FAIL'}\n")
    
    # Test 2: Get directory information
    print(Fore.CYAN + "Test 2: Getting directory information...")
    dir_info = get_directory_info(Config.RESULTS_BASE_DIR)
    if dir_info:
        print(Fore.GREEN + "Directory Information:")
        for key, value in dir_info.items():
            print(f"  {key}: {value}")
    else:
        print(Fore.RED + "No directory information available")
    print()
    
    # Test 3: Test with non-existent directory
    print(Fore.CYAN + "Test 3: Testing with non-existent directory...")
    test_dir = "test_non_existent_dir"
    result = validate_directory_structure(test_dir)
    print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Cleanup test directory
    if os.path.exists(test_dir):
        try:
            os.rmdir(test_dir)
            print(Fore.YELLOW + f"Cleaned up test directory: {test_dir}")
        except:
            print(Fore.YELLOW + f"Could not clean up test directory: {test_dir}")
    print()
    
    # Test 4: Test with invalid path
    print(Fore.CYAN + "Test 4: Testing with invalid path...")
    invalid_path = "/root/invalid_path_test" if os.name != 'nt' else "C:\\invalid_path_test"
    result = validate_directory_structure(invalid_path)
    print(f"Result: {'❌ Expected failure' if not result else '⚠️ Unexpected success'}\n")
    
    print(Fore.GREEN + "=== Directory validation tests completed ===")

if __name__ == "__main__":
    test_directory_validation()
