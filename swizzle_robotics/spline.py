import numpy as np

class QuinticSplineTrajectory:
    """
    Quintic (5th order) polynomial spline generator.
    Guarantees continuous position, velocity, and acceleration (jerk-free)
    boundary conditions for robotic arm manipulation.
    """
    @staticmethod
    def generate(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float, steps: int = 100):
        """
        p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0

        A = np.array([
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3]
        ])
        b = np.array([
            p1 - c0 - c1 * T - c2 * T2,
            v1 - c1 - 2 * c2 * T,
            a1 - 2 * c2
        ])

        c3, c4, c5 = np.linalg.solve(A, b)

        times = np.linspace(0, T, steps)
        pos = c0 + c1 * times + c2 * (times**2) + c3 * (times**3) + c4 * (times**4) + c5 * (times**5)
        vel = c1 + 2 * c2 * times + 3 * c3 * (times**2) + 4 * c4 * (times**3) + 5 * c5 * (times**4)
        acc = 2 * c2 + 6 * c3 * times + 12 * c4 * (times**2) + 20 * c5 * (times**3)
        return times, pos, vel, acc
