import numpy as np
import matplotlib.pyplot as plt

# =========================
# LIF parameters
# =========================
V_REST = -70e-3
V_RESET = -65e-3
V_THRESHOLD = -50e-3
R = 10e6
TAU = 8e-3
REFRACTORY_PERIOD = 2e-3

# =========================
# Simulation setup
# =========================
dt = 1e-4      # 0.1 ms
T = 0.25
time = np.arange(0, T, dt)

# =========================
# Input currents
# =========================
def step_current(t, I0=3.5e-9, t_on=0.2, t_off=0.8):
    return np.where((t >= t_on) & (t <= t_off), I0, 0.0)

def sinusoidal_current(t, I0=2.5e-9, freq=10):
    return I0 * (1 + np.sin(2*np.pi*freq*t))

def slow_noisy_current(t, I_mean=1.9e-9, noise_std=3.0e-9, tau_noise=5e-3):
    """Mean near threshold; strong, correlated noise for irregular spiking."""
    np.random.seed(7)
    white = np.random.normal(0, noise_std, size=len(t))
    filt = np.zeros_like(white)
    for i in range(1, len(white)):
        filt[i] = filt[i-1] + dt/tau_noise * (-filt[i-1] + white[i])
    return I_mean + filt

I_step = step_current(time)
I_sine = sinusoidal_current(time)
I_noise = slow_noisy_current(time)

# =========================
# LIF simulation
# =========================
def lif_neuron(I_t):
    u = V_REST
    u_trace = np.zeros_like(I_t)
    spikes = []
    refr = 0.0
    for i, I in enumerate(I_t):
        if refr > 0:
            u = V_RESET
            refr -= dt
        else:
            du = (-(u - V_REST) + R * I) / TAU
            u += du * dt
            if u >= V_THRESHOLD:
                u = V_RESET
                refr = REFRACTORY_PERIOD
                spikes.append(i * dt)
        u_trace[i] = u
    return u_trace, np.array(spikes)

# =========================
# Run
# =========================
u_step, s_step = lif_neuron(I_step)
u_sine, s_sine = lif_neuron(I_sine)
u_noise, s_noise = lif_neuron(I_noise)

# =========================
# Plot traces + raster
# =========================
fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                         gridspec_kw={'height_ratios':[1,1,1,0.5]})

axes[0].plot(time*1000, u_step*1000)
axes[0].set_title("Neuron 1 – Step current")
axes[0].set_ylabel("V (mV)")

axes[1].plot(time*1000, u_sine*1000)
axes[1].set_title("Neuron 2 – Sinusoidal current")
axes[1].set_ylabel("V (mV)")

axes[2].plot(time*1000, u_noise*1000)
axes[2].set_title("Neuron 3 – Slow noisy current (irregular)")
axes[2].set_ylabel("V (mV)")

for i, spk in enumerate([s_step, s_sine, s_noise]):
    axes[3].vlines(spk*1000, i+0.5, i+1.5)
axes[3].set_ylim(0.5, 3.5)
axes[3].set_yticks([1,2,3])
axes[3].set_yticklabels(["Step","Sine","Noise"])
axes[3].set_xlabel("Time (ms)")
axes[3].set_title("Raster plot of spike times")

plt.tight_layout()
plt.show()

# =========================
# ISI histogram for noisy neuron
# =========================
if len(s_noise) > 2:
    isi = np.diff(s_noise)
    plt.figure(figsize=(6,3))
    plt.hist(isi*1000, bins=20, color="gray")
    plt.xlabel("Inter-spike interval (ms)")
    plt.ylabel("Count")
    plt.title(f"ISI distribution (mean {isi.mean()*1000:.1f} ms, CV {isi.std()/isi.mean():.2f})")
    plt.show()
