# Test package for Synobotix VLA PoC

# Run all tests
# python -m tests

if __name__ == '__main__':
    import unittest
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent), pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
