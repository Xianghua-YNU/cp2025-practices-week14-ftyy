import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    # TODO: 在此实现受迫单摆的ODE方程
    theta, omega = state
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    # TODO: 使用solve_ivp求解受迫单摆方程
    # 提示: 需要调用forced_pendulum_ode函数
   
    t = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(
        lambda t, y: forced_pendulum_ode(t, y, l, g, C, Omega),
        t_span,
        y0,
        t_eval=t,
        rtol=1e-6,
        atol=1e-9
    )
    return sol.t, sol.y[0]

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    if Omega_range is None:
        Omega0 = np.sqrt(g/l)  # 小角度近似下的自然频率
        Omega_range = np.linspace(Omega0/2, 2*Omega0, 50)
    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l, g, C, Omega, t_span, y0)
        steady_idx = t > t_span[0] + (t_span[1]-t_span[0])/2
        amplitude = np.max(np.abs(theta[steady_idx]))
        amplitudes.append(amplitude)

    return Omega_range, amplitudes

def plot_results(t, theta, title):
    """绘制结果"""
    # 此函数已提供完整实现，学生不需要修改
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    # TODO: 调用solve_pendulum和plot_results
    t, theta = solve_pendulum()
    plot_results(t, theta, title='Forced Pendulum Motion')
    
    # 任务2: 探究共振现象
    # TODO: 调用find_resonance并绘制共振曲线
    Omega_range, amplitudes = find_resonance()
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes, 'o-', label='Amplitude')
    plt.title('Resonance Curve')
    plt.axvline(np.sqrt(9.81/0.1), color='r', linestyle='--', 
               label='Natural frequency (small angle)')
    plt.xlabel('Driving Frequency (rad/s)')
    plt.ylabel('Amplitude (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 找到共振频率并绘制共振情况
    max_amplitude_index = np.argmax(amplitudes)
    resonance_frequency = Omega_range[max_amplitude_index]
    print(f"共振频率: (Ω={resonance_frequency:.3f} rad/s)")
    t_res, theta_res = solve_pendulum(Omega=resonance_frequency)
    plot_results(t_res, theta_res, f'Forced Pendulum at Resonance (Ω={resonance_frequency:.3f} rad/s)')

if __name__ == '__main__':
    main()
