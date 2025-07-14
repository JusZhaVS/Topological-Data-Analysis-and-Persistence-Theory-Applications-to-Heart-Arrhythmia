#!/usr/bin/env python3
"""
Test runner for TDA cardiac analysis package.
"""

import unittest
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_tests(verbosity=2):
    """Run all tests in the tests directory."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner function."""
    print("TDA Cardiac Analysis - Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    if not os.path.exists('tda_arrhythmia'):
        print("Error: Please run from the project root directory")
        print(f"Current directory: {current_dir}")
        return False
    
    # Run tests
    success = run_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 50)
    
    return success


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)