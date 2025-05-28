#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def lorenz_system(state, sigma, r, b):
    """
    定义洛伦兹系统方程
    
    参数:
        state: 当前状态向量 [x, y, z]
        sigma, r, b: 系统参数
        
    返回:
        导数向量 [dx/dt, dy/dt, dz/dt]
    """
    # TODO: 实现洛伦兹系统方程 (约3行代码)
    # [STUDENT_CODE_HERE]
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (r - z) - y
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])
    


def solve_lorenz_equations(sigma=10.0, r=28.0, b=8/3,
                          x0=0.1, y0=0.1, z0=0.1,
                          t_span=(0, 50), dt=0.01):
    """
    求解洛伦兹方程
    
    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    # TODO: 使用solve_ivp求解洛伦兹方程 (约3行代码)
    # [STUDENT_CODE_HERE]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lambda t, state: lorenz_system(state, sigma, r, b),
                    t_span, [x0, y0, z0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y
    


def plot_lorenz_attractor(t: np.ndarray, y: np.ndarray):
    """
    绘制洛伦兹吸引子3D图
    """
    # TODO: 实现3D绘图 (约6行代码)
    # [STUDENT_CODE_HERE]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0], y[1], y[2], lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz Attractor')
    plt.tight_layout()
    plt.show()


def compare_initial_conditions(ic1, ic2, t_span=(0, 50), dt=0.01):
    """
    比较不同初始条件的解
    """
    # TODO: 实现初始条件比较 (约10行代码)
    # [STUDENT_CODE_HERE]
    t1, y1 = solve_lorenz_equations(x0=ic1[0], y0=ic1[1], z0=ic1[2], t_span=t_span, dt=dt)
    t2, y2 = solve_lorenz_equations(x0=ic2[0], y0=ic2[1], z0=ic2[2], t_span=t_span, dt=dt)
    plt.figure(figsize=(8, 4))
    plt.plot(t1, y1[0], label='x(t), IC1')
    plt.plot(t2, y2[0], label="x'(t), IC2", linestyle='--')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Effect of Small Change in Initial Condition on x(t)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 可选：绘制轨迹距离随时间变化
    distance = np.sqrt(np.sum((y1 - y2) ** 2, axis=0))
    plt.figure(figsize=(8, 4))
    plt.plot(t1, distance)
    plt.xlabel('t')
    plt.ylabel('distance')
    plt.title('Distance Between Two Trajectories in Phase Space Over Time')
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数，执行所有任务
    """
    # 任务A: 求解洛伦兹方程
    t, y = solve_lorenz_equations()
    
    # 任务B: 绘制洛伦兹吸引子
    plot_lorenz_attractor(t, y)
    
    # 任务C: 比较不同初始条件
    ic1 = (0.1, 0.1, 0.1)
    ic2 = (0.10001, 0.1, 0.1)  # 微小变化
    compare_initial_conditions(ic1, ic2)


if __name__ == '__main__':
    main()
