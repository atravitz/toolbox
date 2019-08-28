"""tests for toolbox.py module."""
import numpy as np
from toolbox import *
import unittest
import numpy.testing as nptest
import os
import matplotlib.pyplot as plt

class PotentialTests(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1.0
        self.sigma = 1.0
        self.R = 0
        self.r_values = np.linspace(1+self.R, 2.5*self.sigma+self.R, 100)
        self.V_true, self.F_true = lj(r=self.r_values,
                            epsilon=self.epsilon,
                            sigma=self.sigma,
                            R=self.R)
    def test_lj(self):
        pass

    def test_force_to_potential(self):
        """Uses a lj potential to test force_to_potential()."""
        # TODO: Come up with reasonable and universal criteria

        V_test = force_to_potential(self.r_values, self.F_true)

        plt.figure(dpi=120)
        plt.plot(self.r_values, self.F_true, 'k--', label='F_true')
        plt.plot(self.r_values, self.V_true, '.', label='V_true')
        plt.plot(self.r_values[1:], V_test,'.', label='V_test')
        plt.legend()
        plt.savefig('test_f2p.png')

        percent_error = np.abs((V_test-self.V_true[1:]))
        self.assertLess(np.max(percent_error), 2)

    def test_potential_to_force(self):
        """Uses a lj potential to test force_to_potential()."""
        # TODO: Come up with reasonable and universal criteria

        F_test = potential_to_force(self.r_values, self.V_true)

        plt.figure(dpi=120)
        plt.plot(self.r_values, self.V_true, 'k--', label='V_true')
        plt.plot(self.r_values, self.F_true, '.', label='F_true')
        plt.plot(self.r_values, F_test,'.', label='F_test')
        plt.savefig('test_p2f.png')

        percent_error = np.abs((F_test-self.F_true))/np.abs(self.F_true)
        self.assertLess(np.max(percent_error), 1)

class CorrTests(unittest.TestCase):
    def test_scalar_sin(self):
        """test autocorr with a sin function """
        x_range = np.linspace(0,3,100)
        sin_x = np.sin(x_range*2*np.pi)
        autocorr(sin_x)

    def test_scalar_random(self):
        random = np.random.standard_normal(size=50)
        ac = autocorr(random)
        self.assertEqual(ac[0], 1.0)

    def test_vec_unit(self):
        unit_vecs = np.array([[0,0,1], [0,0,1]])

    def test_vec_random(self):
        random = np.random.standard_normal(size=(50,3))
        ac = vector_autocorr(random)
        self.assertEqual(ac[0], 1.0)

if __name__ == '__main__':
    unittest.main()