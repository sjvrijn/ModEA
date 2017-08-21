# tests/__main__.py
import unittest

# import your test modules
import tests

# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
for mod in tests.modules_to_test:
    suite.addTests(loader.loadTestsFromModule(mod))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner()
result = runner.run(suite)