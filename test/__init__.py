"""Run all the valid tests."""
import os
import unittest


class ConfigTestCase(unittest.TestCase):
    def create_test_suite(self):
        test_loader = unittest.TestLoader()
        testSuite = test_loader.discover('.', pattern='test_*.py')

        return testSuite

    def runTest(self):
        self.create_test_suite()


if __name__ == '__main__':
    unittest.main()
    os.remove('fpv_store.sqlite')
