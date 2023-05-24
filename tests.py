import unittest

if __name__ == "__main__":
    # unittest.main(verbosity=2)
    # Create a test suite
    test_suite = unittest.TestLoader().discover(".", pattern="test_*.py")

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the test suite
    runner.run(test_suite)
