""" Set of functions to test minimization algorithms. """
import numpy as np

def implemented_functions():
    list_functions = ['Simple1D', 'Rosenbrock', 'Wood', 'Beale',
                      'FreudensteinRoth', 'GoldsteinPrice','Himmelblau',
                      'MullerBrown', 'PowellSingular', 'PowellBadlyScaled',
                      'BrownBadlyScaled', 'ExtendedRosenbrock4',
                      'ExtendedRosenbrock6', 'NoiseMullerBrown',
                      'NoiseRosenbrock', 'NoiseLennardJones']
    return list_functions


class NoiseLennardJones():

    def __init__(self):
        self.name = 'Noise Lennard Jones'
        self.dimensions = 3
        self.global_optimum = np.array([[ 3.93, 3.93, 3.93]])
        self.fglob = -0.187500
        self.bounds = [[1.0, 3.5], [1.0, 3.5]]
        self.starting_point = [np.repeat([3.0],self.dimensions)]
        self.crange = np.linspace(-15,100,200)
        self.crange2 = 100
        self.epsilon = (0.25 * np.ones(self.dimensions))
        self.sigma = (3.5 * np.ones(self.dimensions))
        self.noise_function = 15e-4
        self.noise_deriv = 0.015


    def evaluate(self, x, *args):
        e = 0
        for i in range(0,len(x)):
            e_i = self.epsilon[i] * ((self.sigma[i]/x[i])**12  - (self.sigma[
            i]/x[i])**6)
            e = e + e_i
        noise = np.random.normal(0,np.abs(self.noise_function*e),1)[0]
        return e + noise

    def jacobian(self, x, *args):
        noise = np.random.normal(loc=x,size=self.dimensions)
        der = np.zeros_like(x)
        for i in range(0,len(x)):
            der[i] = self.epsilon[i]*(-12*self.sigma[i]**12/x[i]**13 +
            6*self.sigma[i]**6/x[i]**7)
        return der + self.noise_deriv * noise



class ExtendedRosenbrock100():

    def __init__(self):
        self.name = 'Extended Rosenbrock 100'
        self.dimensions = 100
        self.global_optimum = [np.repeat([0.0],self.dimensions)]
        self.fglob = 0.0
        self.starting_point = [np.repeat([-1.2,1.0],self.dimensions/2)]


    def evaluate(self, x, *args):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


    def jacobian(self, x, *args):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der



class ExtendedRosenbrock4():

    def __init__(self):
        self.name = 'Extended Rosenbrock 4'
        self.dimensions = 4
        self.global_optimum = [np.repeat([0.0],self.dimensions)]
        self.fglob = 0.0
        self.starting_point = [[-1.2,1.0,-1.2,1.0]]


    def evaluate(self, x, *args):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


    def jacobian(self, x, *args):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der


class ExtendedRosenbrock6():

    def __init__(self):
        self.name = 'Extended Rosenbrock 6'
        self.dimensions = 6
        self.global_optimum = [np.repeat([0.0],self.dimensions)]
        self.fglob = 0.0
        self.starting_point = [[-1.2,1.0,-1.2,1.0,-1.2,1.0]]


    def evaluate(self, x, *args):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


    def jacobian(self, x, *args):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der


class NoiseRosenbrock():

    def __init__(self):
        self.name = 'Noise Rosenbrock'
        self.dimensions = 2
        self.bounds = [[0, 4.0], [0, 4.0]]
        self.global_optimum = np.array([[1.0, 1.0]])
        self.fglob = 0.0
        self.starting_point = [[-1.2, 1.0]]
        self.crange = np.linspace(0,1000,100)
        self.crange2 = 50
        self.noise_function = 0.2
        self.noise_deriv = 0.01


    def evaluate(self, x, *args):
        e = (((1.0 - x[0])**2.0) + 100.0 * (x[1] - x[
        0]**2.0)**2.0)
        noise = np.random.normal(0,np.abs(self.noise_function*e),1)[0]
        return e + self.noise_function * noise


    def jacobian(self, x, *args):
        noise = np.random.normal(loc=x,size=self.dimensions)
        der = np.zeros_like(x)
        der[0] = (-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] -x[0]**2.0)
        )
        der[1] = 200 * (x[1] - x[0]**2.0)
        return der + self.noise_deriv * noise


class NoiseMullerBrown():

    def __init__(self):
        self.dimensions = 2
        self.bounds = [[-1.5, 1.2], [-0.5, 2.1]]
        self.global_optimum = np.array([[-0.558, 1.442]])
        self.fglob = -146.700
        self.starting_point = [[-1.5, 1.5]]
        self.name = 'Muller Brown'
        self.crange = np.linspace(-180,100,100)
        self.crange2 = 250
        self.noise_function = 0.03
        self.noise_deriv = 0.50


        # Constants:
        self.A1, self.a1, self.b1, self.c1, self.x01, self.y01 = -200.0, \
        -1.0, 0.0, -10, 1.0, 0.0
        self.A2, self.a2, self.b2, self.c2, self.x02, self.y02 = -100.0, \
        -1.0, 0.0, -10, 0.0, 0.5
        self.A3, self.a3, self.b3, self.c3, self.x03, self.y03 = -170.0, \
        -6.5, 11.0, -6.5, -0.5, 1.5
        self.A4, self.a4, self.b4, self.c4, self.x04, self.y04 = 15.0, 0.7, \
     0.6, 0.7, -1.0, 1.0


    def evaluate(self, x, *args):
        z1 = self.A1 * np.exp((self.a1*(x[0]-self.x01)**2)+(self.b1*(
        x[0]-self.x01)*(x[1]-self.y01))+(self.c1*(
        x[1]-self.y01)**2))
        z2 = self.A2 * np.exp((self.a2*(x[0]-self.x02)**2)+(self.b2*(x[0]-self.x02)*(x[1]-self.y02))+(self.c2*(
      x[1]-self.y02)**2))
        z3 = self.A3 * np.exp((self.a3*(x[0]-self.x03)**2)+(self.b3*(x[0]-self.x03)*(x[1]-self.y03))+(self.c3*(
      x[1]-self.y03)**2))
        z4 = self.A4 * np.exp((self.a4*(x[0]-self.x04)**2)+(self.b4*(x[0]-self.x04)*(x[1]-self.y04))+(self.c4*(
      x[1]-self.y04)**2))
        e = z1+z2+z3+z4
        noise = np.random.normal(0,np.abs(self.noise_function*e),1)[0]
        return e + self.noise_function * noise


    def jacobian(self, x, *args):
        z1 = self.A1 * np.exp((self.a1*(x[0]-self.x01)**2)+(self.b1*(
        x[0]-self.x01)*(x[1]-self.y01))+(self.c1*(
        x[1]-self.y01)**2))
        z2 = self.A2 * np.exp((self.a2*(x[0]-self.x02)**2)+(self.b2*(x[0]-self.x02)*(x[1]-self.y02))+(self.c2*(
      x[1]-self.y02)**2))
        z3 = self.A3 * np.exp((self.a3*(x[0]-self.x03)**2)+(self.b3*(x[0]-self.x03)*(x[1]-self.y03))+(self.c3*(
      x[1]-self.y03)**2))
        z4 = self.A4 * np.exp((self.a4*(x[0]-self.x04)**2)+(self.b4*(x[0]-self.x04)*(x[1]-self.y04))+(self.c4*(
      x[1]-self.y04)**2))
        der = np.zeros_like(x)
        dx1 = z1 * (2*self.a1*(x[0]-self.x01)+self.b1*(x[1]-self.y01))
        dx2 = z2 * (2*self.a2*(x[0]-self.x02)+self.b2*(x[1]-self.y02))
        dx3 = z3 * (2*self.a3*(x[0]-self.x03)+self.b3*(x[1]-self.y03))
        dx4 = z4 * (2*self.a4*(x[0]-self.x04)+self.b4*(x[1]-self.y04))
        dy1 = z1 * (self.b1*(x[0]-self.x01)+2*self.c1*(x[1]-self.y01))
        dy2 = z2 * (self.b2*(x[0]-self.x02)+2*self.c2*(x[1]-self.y02))
        dy3 = z3 * (self.b3*(x[0]-self.x03)+2*self.c3*(x[1]-self.y03))
        dy4 = z4 * (self.b4*(x[0]-self.x04)+2*self.c4*(x[1]-self.y04))
        der[0] = dx1 + dx2 + dx3 + dx4
        der[1] = dy1 + dy2 + dy3 + dy4
        noise = np.zeros_like(x)
        noise[0] = np.random.normal(0,np.abs(self.noise_function*der[0]),1)[0]
        noise[1] = np.random.normal(0,np.abs(self.noise_function*der[1]),1)[0]
        return der + self.noise_deriv * noise


class Himmelblau():

    def __init__(self):
        self.name = 'Himmelblau'
        self.dimensions = 2
        self.bounds = [[-6, 6], [-6, 6]]
        self.global_optimum = np.array([[-3.779310, -3.283186]])
        self.fglob = 0.0
        self.starting_point = [[-0.5, -1.0]]
        self.crange = np.linspace(0,500,100)
        self.crange2 = 250


    def evaluate(self, x, *args):
        return (x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 -7)**2


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*x[0] + 2*x[1]**2 - 14
        der[1] = 2*x[0]**2 + 4*x[1]*(x[0] + x[1]**2 - 7) + 2*x[1] - 22
        return der


class GoldsteinPrice():


    def __init__(self):
        self.name = 'Goldstein Price'
        self.dimensions = 2
        self.bounds = [[-1, 1], [1, -2]]
        self.global_optimum = np.array([[0.0, -1.0]])
        self.fglob = 3.0
        self.starting_point = [[0.0, 0.0]]
        self.crange = np.linspace(-15,1000,100)
        self.crange2 = 250


    def evaluate(self, x, *args):
        return (1 + (1 + x[0] + x[1])**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14
        * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * (30 + (2 * x[0] - 3 * x[
        1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] -  36 * x[0] * x[1]
         + 27 * x[1]**2))


    def jacobian(self, x, *args):
        der = np.zeros_like(x)

        der[0] = (1 + (1 + x[0] + x[1])**2 * (19 - 14 * x[0] + 3 *x[0]**2 -
        14* x[1] + 6* x[0]* x[1] +
       3* x[1]**2))* ((-32 + 24* x[0] - 36* x[1])* (2* x[0] - 3* x[1])**2 +
    4* (2* x[0] - 3* x[1])* (18 - 32* x[0] + 12* x[0]**2 + 48* x[1] - 36* x[
    0] *x[1] +
       27 *x[1]**2)) + ((1 + x[0] + x[1])**2 *(-14 + 6* x[0] + 6* x[1]) +
    2* (1 + x[0] + x[1])* (19 - 14* x[0] + 3* x[0]**2 - 14* x[1] + 6* x[0]* x[
    1] +
       3* x[1]**2))* (30 + (2* x[0] - 3* x[1])**2 *(18 - 32* x[0] + 12* x[0]**2
        + 48* x[1] -
       36* x[0]* x[1] + 27* x[1]**2))

        der[1] = (1 + (1 + x[0] + x[1])**2 * (19 - 14* x[0] + 3* x[0]**2 - 14*
        x[1] + 6 *x[0] * x[1] +
       3* x[1]**2))* ((2* x[0] - 3* x[1])**2 *(48 - 36* x[0] + 54* x[1]) -
    6* (2* x[0] - 3* x[1])* (18 - 32* x[0] + 12* x[0]**2 + 48* x[1] - 36* x[
    0] * x[1] +
       27 * x[1]**2)) + ((1 + x[0] + x[1])**2 * (-14 + 6 * x[0] + 6 * x[1]) +
    2 * (1 + x[0] + x[1]) * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[
    0] * x[1] +
       3 * x[1]**2)) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12
        * x[0]**2 + 48 * x[1] -
       36 * x[0] * x[1] + 27 * x[1]**2))
        return der


class PowellSingular():

    def __init__(self):
        self.dimensions = 4
        self.bounds = [[-4, 5], [-4, 5], [-4, 5], [-4, 5]]
        self.global_optimum = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.fglob = 0.0
        self.starting_point = [[3.0, -1.0, 0.0, 1.0]]
        self.name = 'Powell Singular'


    def evaluate(self, x, *args):
        return (x[0] + 10 * x[1])**2 + (x[1] - 2 * x[2])**4 + 10 * (x[0] - x[
        3])**4 + 5 * (x[2] - x[3])**2


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = 2 * (x[0] + 10 * x[1]) + 40 * (x[0] - x[3])**3
        der[1] = 20 * (x[0] + 10 * x[1]) + 4 * (x[1] - 2 * x[2])**3
        der[2] = -8 * (x[1] - 2 * x[2])**3 + 10 * (x[2] - x[3])
        der[3] = -40 * (x[0] - x[3])**3 - 10 * (x[2] - x[3])
        return der


class Wood():

    def __init__(self):
        self.dimensions = 4
        self.bounds = [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0],
        [100.0,100.0]]
        self.global_optimum = np.array([[1.0, 1.0, 1.0, 1.0]])
        self.fglob = 0.0
        self.starting_point = [[-1.0, 0.0, -1.0, 0.0]]
        self.name = 'Wood'


    def evaluate(self, x, *args):
        return (1.0 - x[0])**2.0 + 100.0 * (-x[0]**2.0 + x[1])**2 + (1.0 - x[
        2])**2.0 + 1.0/10.0 * (x[1] - x[3])**2.0 + 10.0 * (-2.0 + x[1] + x[
        3])**2.0 + 90.0 * (-x[2]**2.0 + x[3])**2.0


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = -2 * (1 - x[0]) - 400 * x[0] * (-x[0]**2 + x[1])
        der[1] = 200 * (-x[0]**2 + x[1]) + (x[1] - x[3])/5 + 20 * (-2 + x[1] +
        x[3])
        der[2] = -2 * (1 - x[2]) - 360 * x[2] * (-x[2]**2 + x[3])
        der[3] = 1/5 * (-x[1] + x[3]) + 20 * (-2 + x[1] + x[3]) + 180 * (-x[
        2]**2 + x[3])
        return der


class Beale():

    def __init__(self):
        self.name = 'Beale'
        self.dimensions = 2
        self.bounds = [[-4.5, 4.5], [-4.0, 4.0]]
        self.global_optimum = np.array([[3.0, 0.5]])
        self.fglob = 0.0
        self.starting_point = [[1.0, 1.0]]
        self.crange = np.linspace(0,50,100)
        self.crange2 = 250


    def evaluate(self, x, *args):
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0] * x[
        1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = (2*x[1] - 2)*(x[1]*x[0] - x[0] + 1.5) + (2*x[1]**2 - 2)*(x[1]**2*x[0] - x[0] + 2.25) + (2*x[1]**3 - 2)*(x[1]**3*x[0] - x[0] + 2.625)
        der[1] = 6*x[1]**2*x[0]*(x[1]**3*x[0] - x[0] + 2.625) + 4*x[1]*x[0]*(x[1]**2*x[0] - x[0] + 2.25) + 2*x[0]*(x[1]*x[0] - x[0] + 1.5)
        return der


class Rosenbrock():

    def __init__(self):
        self.name = 'Rosenbrock'
        self.dimensions = 2
        self.bounds = [[-1.2, 1.0], [-1.2, 1.0]]
        self.global_optimum = np.array([[1.0, 1.0]])
        self.fglob = 0.0
        self.starting_point = [[2.0, 2.0]]
        self.crange = np.linspace(0,1000,100)
        self.crange2 = 50


    def evaluate(self, x, *args):
        return ((1.0 - x[0])**2.0) + 100.0 * (x[1] - x[0]**2.0)**2.0


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] -x[0]**2.0)
        der[1] = 200 * (x[1] - x[0]**2.0)
        return der


class FreudensteinRoth():

    def __init__(self):
        self.name = 'Freudenstein Roth'
        self.dimensions = 2
        self.bounds = [[-20.0, 35.0], [-4.0, 6.0]]
        self.global_optimum = np.array([[11.41, -0.8968]])
        self.fglob = 48.9842
        self.starting_point = [[ 0.5, -2.0]]
        # self.starting_point = [[ 0.5, 4.0 ]]
        self.crange = np.linspace(0,3000,100)
        self.crange2 = 100


    def evaluate(self, x, *args):
        return (-13.0 + x[0] + x[1] * (-2.0 + (5.0 - x[1]) * x[1]))**2.0 + (-29.0
                + x[0] + x[1] * (-14.0 + x[1] * (1.0 + x[1])))**2.0


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = 2.0 * (-13.0 + x[0] + x[1] * (-2.0 + (5.0 - x[1]) * x[1])) + \
                2.0 * (-29.0 + x[0] + x[1] * (-14.0 + x[1] * (1.0 + x[1])))
        der[1] = 2.0 * (-2.0 + (5.0 - 2.0 * x[1]) * x[1] + (5.0 - x[1])* x[1]) * (
                -13.0 + x[0] + x[1] * (-2.0 + (5.0 - x[1]) * x[1])) + 2.0 * (-14.0 +
                 x[1] * (1.0 + x[1]) + x[1] * (1.0 + 2.0 * x[1])) * (-29.0 + x[0] + x[1] * (
                 -14.0 + x[1] * (1.0 + x[1])))
        return der

class MullerBrown():

    def __init__(self):
        self.dimensions = 2
        self.bounds = [[-1.5, 1.2], [-0.5, 2.1]]
        self.global_optimum = np.array([[-0.558, 1.442]])
        self.fglob = -146.700
        self.starting_point = [[-1.0, 1.0]]
        self.name = 'Muller Brown'
        self.crange = np.linspace(-150,100,100)
        self.crange2 = 250

        # Constants:
        self.A1, self.a1, self.b1, self.c1, self.x01, self.y01 = -200.0, \
        -1.0, 0.0, -10, 1.0, 0.0
        self.A2, self.a2, self.b2, self.c2, self.x02, self.y02 = -100.0, \
        -1.0, 0.0, -10, 0.0, 0.5
        self.A3, self.a3, self.b3, self.c3, self.x03, self.y03 = -170.0, \
        -6.5, 11.0, -6.5, -0.5, 1.5
        self.A4, self.a4, self.b4, self.c4, self.x04, self.y04 = 15.0, 0.7, \
     0.6, 0.7, -1.0, 1.0


    def evaluate(self, x, *args):
        z1 = self.A1 * np.exp((self.a1*(x[0]-self.x01)**2)+(self.b1*(
        x[0]-self.x01)*(x[1]-self.y01))+(self.c1*(
        x[1]-self.y01)**2))
        z2 = self.A2 * np.exp((self.a2*(x[0]-self.x02)**2)+(self.b2*(x[0]-self.x02)*(x[1]-self.y02))+(self.c2*(
      x[1]-self.y02)**2))
        z3 = self.A3 * np.exp((self.a3*(x[0]-self.x03)**2)+(self.b3*(x[0]-self.x03)*(x[1]-self.y03))+(self.c3*(
      x[1]-self.y03)**2))
        z4 = self.A4 * np.exp((self.a4*(x[0]-self.x04)**2)+(self.b4*(x[0]-self.x04)*(x[1]-self.y04))+(self.c4*(
      x[1]-self.y04)**2))
        return z1+z2+z3+z4


    def jacobian(self, x, *args):
        z1 = self.A1 * np.exp((self.a1*(x[0]-self.x01)**2)+(self.b1*(
        x[0]-self.x01)*(x[1]-self.y01))+(self.c1*(
        x[1]-self.y01)**2))
        z2 = self.A2 * np.exp((self.a2*(x[0]-self.x02)**2)+(self.b2*(x[0]-self.x02)*(x[1]-self.y02))+(self.c2*(
      x[1]-self.y02)**2))
        z3 = self.A3 * np.exp((self.a3*(x[0]-self.x03)**2)+(self.b3*(x[0]-self.x03)*(x[1]-self.y03))+(self.c3*(
      x[1]-self.y03)**2))
        z4 = self.A4 * np.exp((self.a4*(x[0]-self.x04)**2)+(self.b4*(x[0]-self.x04)*(x[1]-self.y04))+(self.c4*(
      x[1]-self.y04)**2))
        der = np.zeros_like(x)
        dx1 = z1 * (2*self.a1*(x[0]-self.x01)+self.b1*(x[1]-self.y01))
        dx2 = z2 * (2*self.a2*(x[0]-self.x02)+self.b2*(x[1]-self.y02))
        dx3 = z3 * (2*self.a3*(x[0]-self.x03)+self.b3*(x[1]-self.y03))
        dx4 = z4 * (2*self.a4*(x[0]-self.x04)+self.b4*(x[1]-self.y04))
        dy1 = z1 * (self.b1*(x[0]-self.x01)+2*self.c1*(x[1]-self.y01))
        dy2 = z2 * (self.b2*(x[0]-self.x02)+2*self.c2*(x[1]-self.y02))
        dy3 = z3 * (self.b3*(x[0]-self.x03)+2*self.c3*(x[1]-self.y03))
        dy4 = z4 * (self.b4*(x[0]-self.x04)+2*self.c4*(x[1]-self.y04))
        der[0] = dx1 + dx2 + dx3 + dx4
        der[1] = dy1 + dy2 + dy3 + dy4
        return der


class PowellBadlyScaled():

    def __init__(self):
        self.name = 'Powell Badly Scaled'
        self.dimensions = 2
        self.bounds = [[-10e-5, 10e-5], [0.0, 15.0]]
        self.global_optimum = np.array([[1.098e-5, 9.106]])
        self.fglob = 0.0
        self.starting_point = [[0.0, 1.0]]
        self.crange = np.linspace(0,10,100)
        self.crange2 = 250


    def evaluate(self, x, *args):
        return (-(10001.0/10000.0) + np.exp(-x[0]) + np.exp(-x[1]))**2.0 + (-1.0 + 10000.0 * x[0] * x[1])**2.0


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = 20000.0*x[1]*(10000.0*x[0]*x[1] - 1.0)**1.0 - 2.0*(-1.0001 + np.exp(-x[1]) + np.exp(-x[0]))**1.0*np.exp(-x[0])
        der[1] = 20000.0*x[0]*(10000.0*x[0]*x[1] - 1.0)**1.0 - 2.0*(-1.0001 + np.exp(-x[1]) + np.exp(-x[0]))**1.0*np.exp(-x[1])
        return der

class BrownBadlyScaled():

    def __init__(self):
        self.name = 'Brown BadlyScaled'
        self.dimensions = 2
        self.bounds = [[1e5, 2e6], [1e-6, 3e-6]]
        self.global_optimum = np.array([[1.0e+6, 2e-6]])
        self.fglob = 0.0
        self.starting_point = [[ 1.0, 1.0 ]]
        self.crange = np.linspace(0.0,100000000000,100)
        self.crange2 = 100


    def evaluate(self, x, *args):
        return (-1000000.0 + x[0])**2.0 + (-(1.0/500000.0) + x[1])**2.0 + (
                    -2.0 + x[0] * x[1])**2.0


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = 2.0 * (-1000000.0 + x[0]) + 2.0 * x[1] * (-2.0 + x[0] * x[1])
        der[1] = 2.0 * (-(1.0/500000.0) + x[1]) + 2.0 * x[0] * (-2.0 + x[0]
        * x[1])
        return der


class Simple1D():

    def __init__(self):
        self.name = 'Simple 1D'
        self.dimensions = 1
        self.bounds = [[0.1, 6.0]]
        self.global_optimum = np.array([[0.1]])
        self.fglob = 0.0
        self.starting_point = [[0.5]]
        self.crange = np.linspace(0.0, 6.0, 500)
        self.crange2 = 100


    def evaluate(self, x, *args):
        return 0.3+np.sin(x)*np.cos(x)*np.exp(2*x)*x**-2*np.exp(-x)*np.cos(x) * \
        np.sin(x)


    def jacobian(self, x, *args):
        der = np.zeros_like(x)
        der[0] = (2*np.exp(x)*np.cos(x)**3*np.sin(x))/x**2 - \
        (2*np.exp(x)*np.cos(x)**2*np.sin(x)**2)/x**3 + \
        (np.exp(x)*np.cos(x)**2*np.sin(x)**2)/x**2 - \
        (2*np.exp(x)*np.cos(x)*np.sin(x)**3)/x**2
        return der
