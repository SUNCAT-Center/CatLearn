import numpy as np
from catlearn.optimize.io import *
from catlearn.optimize.get_real_values import *
from ase.optimize import BFGS, LBFGS, MDMin, FIRE
from catlearn.optimize.convergence import *
from ase.io.trajectory import TrajectoryWriter
from ase.neb import NEB, SingleCalculatorNEB
from ase.calculators.emt import EMT
from catlearn.optimize.catlearn_ase_calc import CatLearn_ASE

def create_neb_path(ini_img, end_img, n_images, trained_process, constraints,
                    interp, climb_img):
    ini_img.set_calculator(CatLearn_ASE(trained_process=trained_process))
    end_img.set_calculator(CatLearn_ASE(trained_process=trained_process))
    images = [ini_img]
    images += [ini_img.copy() for i in range(n_images)]
    images += [end_img]
    neb_path = NEB(images, climb=climb_img)
    neb_path.interpolate(interp)
    for image in images[1:n_images]:
        image.set_calculator(CatLearn_ASE(trained_process=trained_process))
        image.set_constraint(constraints)
    return neb_path

