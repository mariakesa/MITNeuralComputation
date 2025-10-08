import numpy as np
import matplotlib.pyplot as plt

# =========================
# LIF parameters
# =========================
V_REST = -70e-3           # resting potential (V)
V_RESET = -65e-3          # reset potential (V)
V_THRESHOLD = -50e-3      # firing threshold (V)
R = 10e6                  # membrane resistance (Ohm)
TAU = 8e-3                # membrane time constant (s)
REFRACTORY_PERIOD = 2e-3  # absolute refractory period (s)

# =========================
# Simulation setup
# =========================
dt = 1e-4     # 0.1 ms
T = 1.0       # 1 s
time = np.arange(0, T, dt)

# =========================
# Input currents (A)
# =========================
def step_current(t, I0=3.5e-9, t_on=0.2, t_off=0.8):
    """Step input current"""
    return np.where((t >= t_on) & (t <= t_off), I0, 0.0)

def sinusoidal_current(t, I0=2.5e-9, freq=10):
    """Sinusoidal input current"""
    return I0 * (1 + np.sin(2 * np.pi * freq * t))  # always positive

def noisy_current(t, I0=2.3e-9, noise_std=1e-9):
    """Noisy input current"""
    np.random.seed(0)
    return I0 + np.random.normal(0, noise_std, size=len(t))

I_step = step_current(time)
I_sine = sinusoidal_current(time)
I_noise = noisy_current(time)

# =========================
# LIF simulation
# =========================
def lif_neuron(I_t):
    u = V_REST
    u_trace = np.zeros_like(I_t)
    refractory = 0.0

    for i, I in enumerate(I_t):
        if refractory > 0:
            u = V_RESET
            refractory -= dt
        else:
            du = (-(u - V_REST) + R * I) / TAU
            u += du * dt
            if u >= V_THRESHOLD:
                u = V_RESET
                refractory = REFRACTORY_PERIOD
        u_trace[i] = u
    return u_trace

# =========================
# Simulate neurons
# =========================
u_step = lif_neuron(I_step)
u_sine = lif_neuron(I_sine)
u_noise = lif_neuron(I_noise)

# =========================
# Plot results
# =========================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time*1000, u_step*1000, lw=1.2)
axes[0].set_title("Neuron 1: Step current input (spikes visible)")
axes[0].set_ylabel("Membrane potential (mV)")

axes[1].plot(time*1000, u_sine*1000, lw=1.2)
axes[1].set_title("Neuron 2: Sinusoidal current input (phase-locked spiking)")
axes[1].set_ylabel("Membrane potential (mV)")

axes[2].plot(time*1000, u_noise*1000, lw=1.2)
axes[2].set_title("Neuron 3: Noisy current input (irregular spiking)")
axes[2].set_ylabel("Membrane potential (mV)")
axes[2].set_xlabel("Time (ms)")

plt.tight_layout()
plt.show()
