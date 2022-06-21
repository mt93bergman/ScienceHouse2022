import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import levy_l
from IPython.display import Image
from IPython.core.display import HTML 

def MM_Eval(S,P,kcat,E0,KM, time, ax1, clear_ax=False):

    #times
    step = 0.1
    stop = time
    t_eval = np.arange(0, stop, step)

    F = lambda t, S: -kcat*E0*S/(S+KM)

    sol = solve_ivp(F, [0,stop], [S], t_eval=t_eval)

    #fig1,ax1 = plt.subplots()
    if clear_ax == True:
        ax1.cla()
    ax1.plot(sol.t, sol.y[0], label='Substrate')
    ax1.plot(sol.t, S-sol.y[0], label='Product')
    ax1.set_title('Enzyme Reaction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Concentration (moles/L)')
    ax1.legend()

def MM_2D(S, E0_range, KM_range, intervals = 50):

    #initial conditions
    S = 1.0

    #parameters
    kcat = 1
    term_val = 0.01
    num_intervals = intervals
    [E0_start,E0_end] = E0_range
    [KM_start, KM_end] = KM_range
    E0_step = (E0_end - E0_start)/(num_intervals-1)
    KM_step = (KM_end - KM_start)/(num_intervals-1)

    E0s = np.arange(E0_start, E0_end+E0_step, E0_step)
    KMs = np.arange(KM_start, KM_end+KM_step, KM_step)
    times = np.zeros((num_intervals,num_intervals))

    KM_mesh,E0_mesh = np.meshgrid(KMs, E0s)

    #times
    stop = 1000

    def MM(t,S, E0, KM):
        val = -kcat*E0*S/(S+KM)
        return val

    def terminate(t,y, foo1, foo2):
        return y[0] - term_val

    terminate.terminal=True
    terminate.direction = -1

    E0_index = 0
    for E0 in E0s:
        KM_index = 0
        for KM in KMs:
            sol = solve_ivp(MM, [0,stop], [S], events=terminate, args=(E0, KM))
            times[E0_index, KM_index] = sol.t_events[0][0]
            KM_index += 1
        E0_index += 1

    fig1,ax1 = plt.subplots()
    plt.pcolormesh(E0_mesh, KM_mesh, times, shading='auto')
    plt.xlabel('E0')
    plt.ylabel('KM')
    plt.title('Effect of E0 and KM on time to consume substrate')
    plt.colorbar(label='time (s)')

def MM_Temp(S, T_range, intervals = 100):

    #initial conditions
    S = 1.0

    #parameters
    term_val = 0.01
    E0 = 0.1
    KM = 0.2
    levy_mean = 2.5
    levy_spread = 4

    num_intervals = intervals
    [T_start,T_end] = T_range
    T_step = (T_end - T_start)/(num_intervals-1)

    Ts = np.arange(T_start, T_end+T_step, T_step)
    Ts_shift_scale = (Ts-330)/10
    kcats = levy_l.pdf(Ts_shift_scale,levy_mean,levy_spread)
    times = np.zeros(Ts.shape)

    #times
    stop = 6000

    def MM(t,S, kcat, E0, KM):
        val = -kcat*E0*S/(S+KM)
        return val

    def terminate(t,y, foo1, foo2, foo3):
        return y[0] - term_val

    terminate.terminal=True
    terminate.direction = -1

    index = 0
    for T in Ts:
        sol = solve_ivp(MM, [0,stop], [S], events=terminate, args=(kcats[index], E0, KM))
        if sol.message=='A termination event occurred.':
            times[index] = sol.t_events[0][0]
        else:
            times[index] = stop
        index += 1

    fig1,ax1 = plt.subplots()
    ax1.scatter(Ts, times)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Time to degrade substrate (s)')
    ax1.set_title('Effect of temperature on time to consume substrate')

def KCat_and_T(T_range, num_intervals=50):
    levy_mean = 2.5
    levy_spread = 4

    [T_start,T_end] = T_range
    T_step = (T_end - T_start)/(num_intervals-1)
    T_step = (T_end - T_start)/(num_intervals-1)
    Ts = np.arange(T_start, T_end+T_step, T_step)
    Ts_shift_scale = (Ts-330)/10
    kcats = levy_l.pdf(Ts_shift_scale,levy_mean,levy_spread)

    fig1,ax1 = plt.subplots()
    ax1.scatter(Ts, kcats)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Kcat (mol/s)')
    ax1.set_title('Effect of temperature on kcat')
