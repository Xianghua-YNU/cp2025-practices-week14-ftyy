import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现简谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])
    # 返回一个包含dx/dt和dv/dt的数组


def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现非谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x^3
    dxdt = v
    dvdt = -omega**2 * x**3  # 非谐振子方程
    return np.array([dxdt, dvdt])
    # 返回一个包含dx/dt和dv/dt的数组


def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    # TODO: 实现RK4方法
    k1 = ode_func(state, t, **kwargs)#
    k2 = ode_func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = ode_func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    # TODO: 实现ODE求解器
    t_start, t_end = t_span #   获取时间范围
    num_steps = int((t_end - t_start) / dt) + 1  # 计算时间步数
    t = np.linspace(t_start, t_end, num_steps) # 生成时间点数组
    states = np.zeros((num_steps, len(initial_state))) # 初始化状态数组
    states[0] = initial_state    # 设置初始状态
    # 使用RK4方法逐步求解
    # 对每个时间步进行RK4积分
    for i in range(1, num_steps):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    return t, states
    

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
    plt.figure()
    plt.plot(t, states[:, 0], label='Displacement x(t)')  # 绘制位移随时间变化曲线
    plt.plot(t, states[:, 1], label='Velocity v(t)')      # 绘制速度随时间变化曲线
    plt.xlabel('Time')  # 设置x轴标签
    plt.ylabel('Value')  # 设置y轴标签
    plt.title(title)  # 设置图像标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图像


def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure()  # 新建一个图像窗口
    plt.plot(states[:, 0], states[:, 1], label='Phase trajectory')  # 绘制相空间轨迹
    plt.xlabel('Displacement x')  # 设置x轴标签
    plt.ylabel('Velocity v')  # 设置y轴标签
    plt.title(title)  # 设置图像标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图像


def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    # TODO: 实现周期分析
    """
    分析振动周期。
    """
    x = states[:, 0]  # 获取位移数组
    zero_crossings = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]  # 找到从负到正的过零点
    if len(zero_crossings) < 2:
        return np.nan  # 如果过零点太少，无法估算周期
    periods = np.diff(t[zero_crossings])  # 计算相邻过零点的时间差
    return np.mean(periods)  # 返回平均周期

def main():
    # 设置参数
    omega = 1.0  # 角频率
    t_span = (0, 50)  # 时间范围
    dt = 0.01  # 时间步长

    # 任务1 - 简谐振子的数值求解
    initial_state = np.array([1.0, 0.0])  # 初始条件：x(0)=1, v(0)=0
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)  # 求解微分方程
    plot_time_evolution(t, states, "Harmonic Oscillator: x(0)=1, v(0)=0")  # 绘制时间演化图
    period = analyze_period(t, states)  # 分析周期
    print(f"Harmonic oscillator period (x(0)=1): {period:.4f}")  # 输出周期

    # 任务2 - 振幅对周期的影响分析
    initial_state2 = np.array([2.0, 0.0])  # 初始条件：x(0)=2, v(0)=0
    t2, states2 = solve_ode(harmonic_oscillator_ode, initial_state2, t_span, dt, omega=omega)  # 求解微分方程
    plot_time_evolution(t2, states2, "Harmonic Oscillator: x(0)=2, v(0)=0")  # 绘制时间演化图
    period2 = analyze_period(t2, states2)  # 分析周期
    print(f"Harmonic oscillator period (x(0)=2): {period2:.4f}")  # 输出周期
    print("Harmonic oscillator period is independent of amplitude (isochronism).")  # 输出等时性说明

    # 任务3 - 非谐振子的数值分析
    t3, states3 = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)  # 求解非谐振子
    plot_time_evolution(t3, states3, "Anharmonic Oscillator: x(0)=1, v(0)=0")  # 绘制时间演化图
    period3 = analyze_period(t3, states3)  # 分析周期
    print(f"Anharmonic oscillator period (x(0)=1): {period3:.4f}")  # 输出周期

    t4, states4 = solve_ode(anharmonic_oscillator_ode, initial_state2, t_span, dt, omega=omega)  # 求解非谐振子
    plot_time_evolution(t4, states4, "Anharmonic Oscillator: x(0)=2, v(0)=0")  # 绘制时间演化图
    period4 = analyze_period(t4, states4)  # 分析周期
    print(f"Anharmonic oscillator period (x(0)=2): {period4:.4f}")  # 输出周期
    print("Anharmonic oscillator period increases with amplitude.")  # 输出振幅影响说明

    # 任务4 - 相空间分析
    plot_phase_space(states, "Harmonic Oscillator Phase Space")  # 绘制简谐振子相空间轨迹
    plot_phase_space(states2, "Harmonic Oscillator Phase Space (Large Amplitude)")  # 大振幅简谐振子
    plot_phase_space(states3, "Anharmonic Oscillator Phase Space")  # 非谐振子
    plot_phase_space(states4, "Anharmonic Oscillator Phase Space (Large Amplitude)")  # 大振幅非谐振子
    print("Harmonic oscillator phase space is an ellipse; anharmonic oscillator is a non-elliptical closed curve.")  # 输出相空间特征说明

if __name__ == "__main__":
    main()
