import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def ode_averaged(y, t, eps):
    """Уравнение второго порядка, приведенное к системе уравнений первого порядка
    Усредненная система
    """
    x1, x2 = y
    dydt = [-eps*x1*(1 - 0.5*np.cos(2*x2)),  -eps*np.sin(2*x2)]
    return dydt


def ode_exact(y, t, eps):
    """равнение второго порядка, приведенное к системе уравнений первого порядка
    Точная система"""
    x1, x2 = y
    dydt = [-4*eps*x1*(np.sin(t+x2)*np.cos(t))**2,  -2*eps*(np.sin(2*t+2*x2)*np.cos(t))**2]
    return dydt


def calc_ode(ODE, args, y0, dy0, ts=10, nt=101):
    """Численное решение, полученное функцией odeint пакета scipy"""
    y0 = y0, dy0  # начальные условия
    t = np.linspace(0, ts, nt)  # время
    solution = odeint(ODE, y0, t, args)  # численное решение
    return solution


def plot_solutions(eps, y0, dy0, ts, nt):
    """Построение решений"""
    for y in y0:
        for dy in dy0:
            solution1 = calc_ode(ode_averaged, (eps,), y, dy, ts, nt)   # найти решение усредненной системы
            solution2 = calc_ode(ode_exact, (eps,), y, dy, ts, nt)      # найти решение точной системы
            t = np.linspace(0, ts, nt)
            x1, x2 = x_x(solution1, t)      # выполнить обратную замену переменных
            y1, y2 = x_x(solution2, t)      # выполнить обратную замену переменных
            fig = plt.figure('Epsilon = ' + str(eps))
            plt.xlabel('x(t)')
            plt.ylabel("x'(t)")
            plt.plot(x1, x2, '-r', figure=fig)
            plt.plot(y1, y2, '--b', figure=fig)
            plt.savefig('Solutions with epsilon = ' + str(eps) + '.jpeg')


def plot_phase_portrait(size,ODE):
    """Построение фазового портрета"""
    y1 = np.linspace(-size, size, 20)
    y2 = np.linspace(-size, size, 20)
    Y1, Y2 = np.meshgrid(y1, y2)
    t = 0
    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
    NI, NJ = Y1.shape
    for i in range(NI):
        for j in range(NJ):
            x = Y1[i, j]
            y = Y2[i, j]
            theta = np.sqrt(x**2+y**2)
            fi = np.math.atan2(y, x)
            yprime = ODE([theta, fi], t, eps)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]
    if ODE == ode_averaged:
        color = 'r'
    else:
        color = 'b'
    plt.quiver(Y1, Y2, u * np.cos(v), u * np.sin(v), color=color)


def plot(ODE, eps):
    plt.figure(ODE.__name__+' with epsilon = ' + str(eps))
    plot_phase_portrait(5, ODE)  # построить фазовый портрет
    plt.xlabel('x(t)')
    plt.ylabel("x'(t)")
    plt.savefig(ODE.__name__+' phase portrait  with epsilon = ' + str(eps)+'.jpeg')


def x_x(solution, t):
    x1 = solution[:, 0] * np.cos(t + solution[:, 1])
    x2 = -solution[:, 0] * np.sin(t + solution[:, 1])
    return x1, x2


def error(epsilons):
    errors = []
    fig = plt.figure('errors')
    for eps in epsilons[::-1]:
        s1 = calc_ode(ode_exact, (eps, ), 1, 0)
        s2 = calc_ode(ode_averaged, (eps, ), 1, 0)
        x = np.max(np.abs(s2 - s1))
        errors.append(x)
    plt.plot(epsilons[::-1], errors, figure=fig)
    plt.xlabel('epsilon')
    plt.ylabel('error')
    plt.savefig('error.jpeg')


if __name__ == '__main__':
    # список эпсилон
    epsilons = [0.5, 0.1, 0.01, 0.001, 0.0]
    # начальные условия
    # можно задать списком
    y0 = [1]
    dy0 = [0]
    errors = []
    for eps in epsilons[::-1]:
        plot_solutions(eps, y0, dy0, 10, 100)   # построить график решений
        plot(ode_exact, eps)        # построить фазовый портрет для точной системы
        plot(ode_averaged, eps)     # построить фазовый портрет для усредненной системы

    # получить максимальные погрешности
    error(epsilons[::-1])
    plt.show()
