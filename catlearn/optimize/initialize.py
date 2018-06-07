import numpy as np
from catlearn.optimize.io import *
from catlearn.optimize.get_real_values import *
from ase.optimize import BFGS, LBFGS, MDMin, FIRE, QuasiNewton
from catlearn.optimize.convergence import *
from ase.io.trajectory import TrajectoryWriter
from ase.neb import NEB, SingleCalculatorNEB
from catlearn.optimize.catlearn_ase_calc import CatLearn_ASE
from catlearn.optimize.get_real_values import *
from ase.io import read, write
from catlearn.optimize.constraints import *


def initialize(self):
    """Evaluates the "real" function for a given initial guess and then it
    will obtain the function value of a second guessed
    point originated using CG theory. This function is exclusively
    called when the optimization is initialized and the user has not
    provided any trained data.
    """

    if len(self.list_targets) == 1:

        if self.jac is False:
            alpha = np.random.normal(loc=0.0, scale=self.i_step,
                                     size=np.shape(self.list_train[0]))
            ini_train = [self.list_train[-1] - alpha]

        if self.jac is True:
            alpha = self.i_step + np.zeros_like(self.list_train[0])
            ini_train = [self.list_train[-1] - alpha *
            self.list_gradients[-1]]

        if self.i_ase_step:
            opt = eval(self.i_ase_step)(self.ase_ini).run(fmax=self.fmax,
                                                          steps=1)
            ini_train = [self.ase_ini.get_positions().flatten()]

        self.list_train = np.append(self.list_train, ini_train, axis=0)
        self.list_targets = np.append(self.list_targets,
                                      get_energy_catlearn(self))
        self.list_targets = np.reshape(self.list_targets,
                                  (len(self.list_targets), 1))
        if self.jac is True:
            self.list_gradients = np.append(
                                        self.list_gradients,
                                        -get_forces_catlearn(self).flatten())
            self.list_gradients = np.reshape(self.list_gradients,
                                             (len(self.list_targets),
                                             np.shape(self.list_train)[1])
                                             )
        self.feval = len(self.list_targets)

        if self.ase:
            molec_writer = TrajectoryWriter('./' + str(self.filename) +
                                                '_catlearn.traj', mode='a')
            molec_writer.write(self.ase_ini)
        converged(self)
        print_info(self)

    if len(self.list_targets) == 0:
        self.list_targets = [np.append(self.list_targets,
                             get_energy_catlearn(self))]
        if self.jac is True:
            self.list_gradients = [np.append(self.list_gradients,
                                   -get_forces_catlearn(self).flatten())]
        self.feval = len(self.list_targets)
        if self.ase:
            molec_writer = TrajectoryWriter('./' + str(self.filename) +
                                                '_catlearn.traj', mode='a')
            molec_writer.write(self.ase_ini)
        converged(self)
        print_info(self)