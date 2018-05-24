import numpy as np
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.io.trajectory import Trajectory

from ase.io import write
from ase.constraints import Filter
from catlearn.optimize.convergence import converged
from prettytable import PrettyTable
import glob, os


def print_info(self):

    def _start_table(self):
        if not self.jac:
            self.table_results = PrettyTable(['Method', 'Iterations',
                                              'Func. evaluations', 'Function '
                                               'value', 'e_diff',
                                               'Converged?'])
        if self.jac:
            self.table_results = PrettyTable(['Method', 'Iterations',
                                              'Func. evaluations', 'Function '
                                              'value', 'fmax', 'Converged?'])

    if self.iter == 0 and self.feval == 1:
        _start_table(self)

    if self.iter == 1 and self.feval > 1:
        if self.start_mode is 'dict':
            _start_table(self)
            for i in range(0, len(self.list_targets)-1):
                if self.jac:
                    self.table_results.add_row(['Previous', self.iter-1,
                                               i+1, self.list_targets[i][0],
                                               self.list_fmax[i], converged(
                                               self)])
                if not self.jac:
                    diff_energ = ['-']
                    if i != 0:
                        diff_energ = self.list_targets[i-1]-self.list_targets[i]
                    self.table_results.add_row(['Previous', self.iter-1,
                                                i+1, self.list_targets[i][0],
                                                diff_energ, converged(self)])

        if self.start_mode is 'trajectory':
            _start_table(self)
            for i in range(0, len(self.list_targets)-1):
                self.table_results.add_row(['Traj. ASE', self.iter-1, i+1,
                                           self.list_targets[i][0],
                                           self.list_fmax[i][0], converged(
                                           self)])

    if self.iter == 0:
        if self.feval == 1:
            if not self.jac:
                self.table_results.add_row(['Eval.', self.iter,
                                           len(self.list_targets),
                                           self.list_targets[-1][0], '-',
                                           converged(self)])
            if self.jac:
                self.table_results.add_row(['Eval.', self.iter,
                                            len(self.list_targets),
                                            self.list_targets[-1][0],
                                            self.max_abs_forces,
                                            converged(self)])

        if self.feval == 2:
            if not self.i_ase_step:
                if not self.jac:
                    self.table_results.add_row(['LineSearch', self.iter,
                                                len(self.list_targets),
                                                self.list_targets[-1][0], '-',
                                                converged(self)])
                if self.jac:
                    self.table_results.add_row(['LineSearch', self.iter,
                                                len(self.list_targets),
                                                self.list_targets[-1][0],
                                                self.max_abs_forces,
                                                converged(self)])
            if self.i_ase_step:
                if not self.jac:
                    self.table_results.add_row([self.i_ase_step, self.iter,
                                                len(self.list_targets),
                                                self.list_targets[-1][0],
                                                self.e_diff, converged(self)])
                if self.jac:
                    self.table_results.add_row([self.i_ase_step, self.iter,
                                                len(self.list_targets),
                                                self.list_targets[-1][0],
                                                self.max_abs_forces,
                                                converged(self)])

    if self.iter > 0:
        if not self.jac:
            self.table_results.add_row(['CatLearn', self.iter,
                                        len(self.list_targets),
                                        self.list_targets[-1][0],
                                        self.e_diff, converged(self)])
        if self.jac:
            self.table_results.add_row(['CatLearn', self.iter,
                                        len(self.list_targets),
                                        self.list_targets[-1][0],
                                        self.max_abs_forces, converged(self)])
    print(self.table_results)
    f_print = open(str(self.filename)+'_convergence_catlearn.txt', 'w')
    f_print.write(str(self.table_results))
    f_print.close()


def print_info_ml(self):
    if self.iter == 0: # Print header
        self.table_ml_results = PrettyTable(['CatLearn iteration', 'Minimizer',
                                             'ML func. evaluations',
                                             'Suggested point (x)',
                                             'Predicted value', 'Converged?'])
    self.table_ml_results.add_row([self.iter, self.ml_algo,
                                   self.ml_feval_pred_mean,
                                   self.interesting_point,
                                   self.ml_f_min_pred_mean,
                                   self.ml_convergence])
    f_ml_print = open(str(self.filename)+'_convergence_ml.txt', 'w')
    f_ml_print.write(str(self.table_ml_results))
    f_ml_print.close()


def store_results(self):
    if not self.jac:
        self.results = {'train': self.list_train,
                        'targets': self.list_targets,
                        'iterations': self.iter,
                        'f_eval': len(self.list_targets),
                        'converged': converged(self)}
    if self.jac:
        self.results = {'train': self.list_train,
                        'targets': self.list_targets,
                        'gradients': self.list_gradients,
                        'forces': -self.list_gradients,
                        'iterations': self.iter,
                        'f_eval': len(self.list_targets),
                        'converged': converged(self)
                 }
    f_res = open(str(self.filename)+"_data.txt","w")
    f_res.write(str(self.results))
    f_res.write(str(dict))


def backup_old_calcs(filename):
    if os.path.isfile("./" + filename + "_catlearn.traj"):
        i = 1
        n_backup = str('{:05d}'.format(i))
        while os.path.isfile("./" + filename + "_catlearn_old" + n_backup +
                             ".traj"):
            i += 1
            n_backup = str('{:05d}'.format(i))
        os.rename("./" + filename + "_catlearn.traj", "./" + filename +
                  "_catlearn_old" + n_backup + ".traj")

    if os.path.isfile("./" + filename + "_convergence_catlearn.txt"):
        j = 1
        n_backup = str('{:05d}'.format(j))
        while os.path.isfile("./" + filename + "_convergence_catlearn_old" +
                             n_backup + ".txt"):
            j += 1
            n_backup = str('{:05d}'.format(j))
        os.rename("./" + filename + "_convergence_catlearn.txt", "./" +
        filename +
                  "_convergence_catlearn_old" + n_backup + ".txt")



