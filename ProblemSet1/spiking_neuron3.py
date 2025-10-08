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
# LIF simulation (records spikes)
# =========================
def lif_neuron(I_t):
    u = V_REST
    u_trace = np.zeros_like(I_t)
    spike_times = []
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
                spike_times.append(i * dt)  # record time in seconds
        u_trace[i] = u

    return u_trace, np.array(spike_times)

# =========================
# Run simulations
# =========================
u_step, spikes_step = lif_neuron(I_step)
u_sine, spikes_sine = lif_neuron(I_sine)
u_noise, spikes_noise = lif_neuron(I_noise)

# =========================
# Plot results
# =========================
fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1, 0.5]})

# Membrane potential traces
axes[0].plot(time*1000, u_step*1000, lw=1.2)
axes[0].set_title("Neuron 1: Step current input")
axes[0].set_ylabel("V (mV)")

axes[1].plot(time*1000, u_sine*1000, lw=1.2)
axes[1].set_title("Neuron 2: Sinusoidal current input")
axes[1].set_ylabel("V (mV)")

axes[2].plot(time*1000, u_noise*1000, lw=1.2)
axes[2].set_title("Neuron 3: Noisy current input")
axes[2].set_ylabel("V (mV)")

# Raster plot
for neuron_idx, spike_times in enumerate([spikes_step, spikes_sine, spikes_noise]):
    axes[3].vlines(spike_times*1000, neuron_idx+0.5, neuron_idx+1.5, lw=1)
axes[3].set_ylim(0.5, 3.5)
axes[3].set_xlim(0, T*1000)
axes[3].set_yticks([1, 2, 3])
axes[3].set_yticklabels(['Step', 'Sine', 'Noise'])
axes[3].set_xlabel("Time (ms)")
axes[3].set_title("Raster plot of spike times")

plt.tight_layout()
plt.show()
