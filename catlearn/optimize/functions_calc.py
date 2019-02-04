import numpy as np
from ase.calculators.calculator import Calculator


class MultiModal(Calculator):

    """MultiModal potential."""

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = self.atoms.get_positions()[0][0:2]

        def multim(x, y, A, B, xmin, ymin):

            # Function Value:
            e_1 = np.exp(-(x - xmin)**2 / B**2)
            e_2 = np.exp(-(y - ymin)**2 / B**2)
            f = -A * e_1 * e_2

            # Gradient dx:
            e_1_dx = -2 * (x - xmin) / B**2 * e_1
            e_2_dx = 0.0

            dx = -A * (e_1_dx * e_2 + e_1 * e_2_dx)

            # Gradient dy:
            e_1_dy = 0.0
            e_2_dy = -2 * (y - ymin) / B**2 * e_2

            dy = -A * (e_1_dy * e_2 + e_1 * e_2_dy)

            return [f, dx, dy]

        list_A = [14.3, 13., 14.5, 13.5, 12.5, 14.0, 13.5, 14.2, 13.0] #
        # Shallow.
        list_B = [0.9, 0.78, 0.81, 0.76, 0.8, 0.79, 0.76, 0.74, 0.79]  # Width.
        list_xmin = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0]
        list_ymin = [0.0, 2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0,]

        e = []
        grad_x = []
        grad_y = []
        for i in range(0, len(list_A)):
            f_value, dx, dy = multim(x[0], x[1],
                                     list_A[i], list_B[i],
                                     list_xmin[i], list_ymin[i])
            e.append(f_value)
            grad_x.append(dx)
            grad_y.append(dy)

        # Energy:
        energy = np.sum(e)

        # Forces:
        fx = np.sum(grad_x)
        fy = np.sum(grad_y)
        fz = 0.0

        forces[0][0] = -fx
        forces[0][1] = -fy
        forces[0][2] = -fz

        self.results['energy'] = energy
        self.results['forces'] = forces


class MullerBrown(Calculator):

    """Muller-brown potential."""

    implemented_properties = ['energy', 'forces']
    default_parameters = {'p1': [-200.0, -1.0,  0.0, -10.0,  1.0, 0.0],
                          'p2': [-100.0, -1.0,  0.0, -10.0,  0.0, 0.5],
                          'p3': [-170.0, -6.5, 11.0,  -6.5, -0.5, 1.5],
                          'p4': [15.0,  0.7,  0.6,   0.7, -1.0, 1.0]}
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = self.atoms.get_positions()[0][0]
        y = self.atoms.get_positions()[0][1]
        z = self.atoms.get_positions()[0][2]

        # Energies
        z1 = self.parameters.p1[0] * np.exp((self.parameters.p1[1]*(
        x-self.parameters.p1[4])**2)+(self.parameters.p1[2]*(
        x-self.parameters.p1[4])*(
        y-self.parameters.p1[5]))+(self.parameters.p1[3]*(
        y-self.parameters.p1[5])**2))

        z2 = self.parameters.p2[0] * np.exp((self.parameters.p2[1]*(
        x-self.parameters.p2[4])**2)+(self.parameters.p2[2]*(
        x-self.parameters.p2[4])*(
        y-self.parameters.p2[5]))+(self.parameters.p2[3]*(
        y-self.parameters.p2[5])**2))

        z3 = self.parameters.p3[0] * np.exp((self.parameters.p3[1]*(
        x-self.parameters.p3[4])**2)+(self.parameters.p3[2]*(
        x-self.parameters.p3[4])*(
        y-self.parameters.p3[5]))+(self.parameters.p3[3]*(
        y-self.parameters.p3[5])**2))

        z4 = self.parameters.p4[0] * np.exp((self.parameters.p4[1]*(
        x-self.parameters.p4[4])**2)+(self.parameters.p4[2]*(
        x-self.parameters.p4[4])*(
        y-self.parameters.p4[5]))+(self.parameters.p4[3]*(
        y-self.parameters.p4[5])**2))

        energy = (z1+z2+z3+z4)

        # Forces
        dx1 = z1 * (2*self.parameters.p1[1]*(x-self.parameters.p1[
        4])+self.parameters.p1[2]*(
        y-self.parameters.p1[5]))

        dx2 = z2 * (2*self.parameters.p2[1]*(x-self.parameters.p2[
        4])+self.parameters.p2[2]*(
        y-self.parameters.p2[5]))

        dx3 = z3 * (2*self.parameters.p3[1]*(x-self.parameters.p3[
        4])+self.parameters.p3[2]*(
        y-self.parameters.p3[5]))

        dx4 = z4 * (2*self.parameters.p4[1]*(x-self.parameters.p4[
        4])+self.parameters.p4[2]*(
        y-self.parameters.p4[5]))

        dy1 = z1 * (self.parameters.p1[2]*(x-self.parameters.p1[
        4])+2*self.parameters.p1[3]*(
        y-self.parameters.p1[5]))

        dy2 = z2 * (self.parameters.p2[2]*(x-self.parameters.p2[
        4])+2*self.parameters.p2[3]*(
        y-self.parameters.p2[5]))

        dy3 = z3 * (self.parameters.p3[2]*(x-self.parameters.p3[
        4])+2*self.parameters.p3[3]*(
        y-self.parameters.p3[5]))

        dy4 = z4 * (self.parameters.p4[2]*(x-self.parameters.p4[
        4])+2*self.parameters.p4[3]*(
        y-self.parameters.p4[5]))

        Fx = dx1 + dx2 + dx3 + dx4
        Fy = dy1 + dy2 + dy3 + dy4
        Fz = 0.0

        forces[0][0] = -Fx
        forces[0][1] = -Fy
        forces[0][2] = -Fz

        self.results['energy'] = energy / 100.0
        self.results['forces'] = forces / 100.0


class GoldsteinPrice(Calculator):

    """GoldsteinPrice potential."""

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = [self.atoms.get_positions()[0][0], self.atoms.get_positions()[
        0][1]]

        energy = (1 + (1 + x[0] + x[1])**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14
                 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * (30 + (2 * x[0]
                 - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1]
                  - 36 * x[0] * x[1] + 27 * x[1]**2))

        fx = (1 + (1 + x[0] + x[1])**2 * (19 - 14 * x[0] + 3 * x[0]**2 -
              14 * x[1] + 6* x[0] * x[1] + 3 * x[1]**2)) * ((-32 + 24 * x[0]
              - 36 * x[1]) * (2* x[0] - 3* x[1])**2 + 4* (2* x[0] - 3* x[
              1]) * (18 - 32 * x[0] + 12* x[0]**2 + 48* x[1] - 36 * x[0] *
              x[1] + 27 *x[1]**2)) + ((1 + x[0] + x[1])**2 *(-14 + 6* x[0]
              + 6* x[1]) + 2* (1 + x[0] + x[1])* (19 - 14* x[0] + 3* x[
              0]**2 - 14* x[1] + 6* x[0]* x[1] + 3* x[1]**2))* (30 + (2* x[
              0] - 3* x[1])**2 *(18 - 32* x[0] + 12* x[0]**2 + 48* x[1] -
              36* x[0]* x[1] + 27* x[1]**2))

        fy = (1 + (1 + x[0] + x[1])**2 * (19 - 14* x[0] + 3* x[0]**2 - 14*
              x[1] + 6 *x[0] * x[1] + 3* x[1]**2))* ((2* x[0] - 3* x[1])**2
              *(48 - 36* x[0] + 54* x[1]) - 6* (2* x[0] - 3* x[1])* (18 -
              32* x[0] + 12* x[0]**2 + 48* x[1] - 36* x[0] * x[1] + 27 * x[
              1]**2)) + ((1 + x[0] + x[1])**2 * (-14 + 6 * x[0] + 6 * x[1]) +
              2 * (1 + x[0] + x[1]) * (19 - 14 * x[0] + 3 * x[0]**2 - 14 *
              x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * (30 + (2 * x[0] - 3 *
               x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 *
               x[0] * x[1] + 27 * x[1]**2))
        fz = 0.0

        forces[0][0] = -fx
        forces[0][1] = -fy
        forces[0][2] = -fz

        self.results['energy'] = 0.01 * energy
        self.results['forces'] = 0.01 * forces


class Himmelblau(Calculator):

    """Himmelblau potential."""

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = self.atoms.get_positions()[0][0:2]

        energy = (x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 -7)**2

        fx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*x[0] + 2*x[1]**2 - 14

        fy = 2*x[0]**2 + 4*x[1]*(x[0] + x[1]**2 - 7) + 2*x[1] - 22
        fz = 0.0

        forces[0][0] = -fx
        forces[0][1] = -fy
        forces[0][2] = -fz

        self.results['energy'] = 0.05 * energy
        self.results['forces'] = 0.05 * forces


class ModifiedHimmelblau(Calculator):

    """Himmelblau potential."""

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = [self.atoms.get_positions()[0][0], self.atoms.get_positions()[
        0][1]]

        energy = (x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 -7)**2 + 0.5 * x[
                  0] + x[1]

        fx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*x[0] + 2*x[1]**2 - 14 + 0.5

        fy = 2*x[0]**2 + 4*x[1]*(x[0] + x[1]**2 - 7) + 2*x[1] - 22 + 1.0

        fz = 0.0

        forces[0][0] = -fx
        forces[0][1] = -fy
        forces[0][2] = -fz

        self.results['energy'] = 0.05 * energy
        self.results['forces'] = 0.05 * forces

class NoiseHimmelblau(Calculator):

    """NoiseHimmelblau potential."""

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = [self.atoms.get_positions()[0][0], self.atoms.get_positions()[
        0][1]]

        energy = (x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 -7)**2 + 0.5 * x[
                  0] + x[1]

        fx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*x[0] + 2*x[1]**2 - 14 + 0.5

        fy = 2*x[0]**2 + 4*x[1]*(x[0] + x[1]**2 - 7) + 2*x[1] - 22 + 1.0

        fz = 0.0

        forces[0][0] = -fx
        forces[0][1] = -fy
        forces[0][2] = -fz


        noise_energy = energy * np.random.normal(scale=0.4)
        self.results['energy'] = (energy + noise_energy)
        noise_force = forces * np.random.normal(scale=0.90)
        self.results['forces'] = (forces + noise_force)


class Rosenbrock(Calculator):

    """Himmelblau potential."""

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy','forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.energy = 0.0
        self.forces = np.zeros((len(self.atoms), 3))
        forces = self.forces

        x = [self.atoms.get_positions()[0][0], self.atoms.get_positions()[
        0][1]]

        energy = 0.01 * (((1.0 - x[0])**2.0) + 100.0 * (x[1] - x[0]**2.0)**2.0)

        fx = 0.01 * (-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] -x[0]**2.0))

        fy = 0.01 * (200 * (x[1] - x[0]**2.0))

        fz = 0.0

        forces[0][0] = -fx
        forces[0][1] = -fy
        forces[0][2] = -fz

        self.results['energy'] = 0.01 * energy
        self.results['forces'] = 0.01 * forces