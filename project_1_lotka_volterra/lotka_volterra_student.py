#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[樊一川]
学号：[20221050017]
完成日期：[2025/5/28]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float, gamma: float, delta: float) -> np.ndarray:
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])

def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    return t, y

def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + k2) / 2
    return t, y

def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1/2, t[i] + dt/2, *args)
        k3 = dt * f(y[i] + k2/2, t[i] + dt/2, *args)
        k4 = dt * f(y[i] + k3, t[i] + dt, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0_vec = np.array([x0, y0])
    t, sol = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x = sol[:, 0]
    y = sol[:, 1]
    return t, x, y

def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float], dt: float) -> dict:
    y0_vec = np.array([x0, y0])
    t_euler, sol_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    t_ieuler, sol_ieuler = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    t_rk4, sol_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    # 误差分析（与RK4对比）
    euler_error = np.max(np.abs(sol_euler - sol_rk4))
    ieuler_error = np.max(np.abs(sol_ieuler - sol_rk4))
    print(f"欧拉法最大误差: {euler_error:.4e}")
    print(f"改进欧拉法最大误差: {ieuler_error:.4e}")
    results = {
        'euler': {'t': t_euler, 'x': sol_euler[:, 0], 'y': sol_euler[:, 1]},
        'improved_euler': {'t': t_ieuler, 'x': sol_ieuler[:, 0], 'y': sol_ieuler[:, 1]},
        'rk4': {'t': t_rk4, 'x': sol_rk4[:, 0], 'y': sol_rk4[:, 1]}
    }
    return results

def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str = "Lotka-Volterra种群动力学") -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, x, label='猎物 x')
    plt.plot(t, y, label='捕食者 y')
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('时间序列')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.xlabel('猎物 x')
    plt.ylabel('捕食者 y')
    plt.title('相空间轨迹')
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_method_comparison(results: dict) -> None:
    methods = ['euler', 'improved_euler', 'rk4']
    titles = ['欧拉法', '改进欧拉法', '四阶Runge-Kutta法']
    plt.figure(figsize=(18, 8))
    for i, method in enumerate(methods):
        plt.subplot(2, 3, i+1)
        plt.plot(results[method]['t'], results[method]['x'], label='猎物 x')
        plt.plot(results[method]['t'], results[method]['y'], label='捕食者 y')
        plt.xlabel('时间')
        plt.ylabel('数量')
        plt.title(f"{titles[i]} 时间序列")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 3, i+4)
        plt.plot(results[method]['x'], results[method]['y'])
        plt.xlabel('猎物 x')
        plt.ylabel('捕食者 y')
        plt.title(f"{titles[i]} 相空间")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_parameters() -> None:
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01
    # 不同初始条件
    initial_conditions = [(2.0, 2.0), (3.0, 1.0), (1.0, 3.0)]
    plt.figure(figsize=(12, 5))
    for x0, y0 in initial_conditions:
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plt.plot(x, y, label=f"x0={x0}, y0={y0}")
    plt.xlabel('猎物 x')
    plt.ylabel('捕食者 y')
    plt.title('不同初始条件下的相空间轨迹')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 不同参数影响
    param_sets = [
        (1.0, 0.5, 0.5, 2.0),
        (1.2, 0.5, 0.5, 2.0),
        (1.0, 0.7, 0.5, 2.0),
        (1.0, 0.5, 0.7, 2.0),
        (1.0, 0.5, 0.5, 2.5)
    ]
    plt.figure(figsize=(12, 5))
    for params in param_sets:
        alpha, beta, gamma, delta = params
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, 2.0, 2.0, t_span, dt)
        plt.plot(x, y, label=f"α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    plt.xlabel('猎物 x')
    plt.ylabel('捕食者 y')
    plt.title('不同参数下的相空间轨迹')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 守恒量验证
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    H = gamma * x - delta * np.log(x) + beta * y - alpha * np.log(y)
    plt.figure(figsize=(8, 4))
    plt.plot(t, H)
    plt.xlabel('时间')
    plt.ylabel('守恒量 H')
    plt.title('守恒量随时间的变化')
    plt.grid(True)
    plt.show()

def main():
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")
    try:
        print("\n1. 使用4阶龙格-库塔法求解...")
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_population_dynamics(t, x, y)
        print("\n2. 比较不同数值方法...")
        results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_method_comparison(results)
        print("\n3. 分析参数影响...")
        analyze_parameters()
        print("\n4. 数值结果统计:")
        print(f"最大猎物数量: {np.max(x):.2f}, 最小猎物数量: {np.min(x):.2f}")
        print(f"最大捕食者数量: {np.max(y):.2f}, 最小捕食者数量: {np.min(y):.2f}")
    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请完成相应函数的实现后再运行主程序。")

if __name__ == "__main__":
    main()
