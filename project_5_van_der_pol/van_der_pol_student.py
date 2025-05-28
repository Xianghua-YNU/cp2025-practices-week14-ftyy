import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    if not isinstance(state, np.ndarray) or len(state) != 2:
        raise ValueError("Expected 'state' to be a NumPy array with two elements [x, v].")
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = ode_func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    states = np.zeros((len(t), len(initial_state)))
    states[0] = initial_state
    for i in range(len(t) - 1):
        states[i+1] = rk4_step(ode_func, states[i], t[i], dt, **kwargs)
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(t, states[:, 0], label='x (位置)')
    plt.plot(t, states[:, 1], label='v (速度)')
    plt.xlabel('时间 t')
    plt.ylabel('状态')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('x (位置)')
    plt.ylabel('v (速度)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    # 振幅：x的最大绝对值
    amplitude = np.max(np.abs(states[:, 0]))
    # 周期：找到x过零点的时刻，计算周期
    x = states[:, 0]
    zero_crossings = np.where(np.diff(np.sign(x)) > 0)[0]
    if len(zero_crossings) > 1:
        periods = np.diff(zero_crossings)
        # 用步长dt估算周期
        period = np.mean(periods)
    else:
        period = np.nan
    return amplitude, period

def main():
    mu = 1.0
    omega = 1.0
    t_span = (0, 20)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])

    # 任务1 - 基本实现
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f"van der Pol 振子时间演化 (mu={mu})")

    # 任务2 - 参数影响分析
    for mu_val in [0.5, 1.0, 2.0, 5.0]:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu_val, omega=omega)
        plot_time_evolution(t, states, f"van der Pol 振子时间演化 (mu={mu_val})")
        plot_phase_space(states, f"van der Pol 相空间 (mu={mu_val})")
        amp, per = analyze_limit_cycle(states)
        print(f"mu={mu_val}: 振幅≈{amp:.3f}, 周期(步数)≈{per:.1f}")

    # 任务3 - 相空间分析
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_phase_space(states, f"van der Pol 相空间 (mu={mu})")
    amp, per = analyze_limit_cycle(states)
    print(f"极限环特征：振幅≈{amp:.3f}, 周期(步数)≈{per:.1f}")

    # 任务4 - 能量分析
    energies = np.array([calculate_energy(s, omega=omega) for s in states])
    plt.figure(figsize=(8, 4))
    plt.plot(t, energies)
    plt.xlabel('时间 t')
    plt.ylabel('能量 E')
    plt.title('能量随时间的变化')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
