import numpy as np
import matplotlib.pyplot as plt

V_REST = -70*10**-3#mV
V_RESET = -65*10**-3#mV
FIRING_THRESHOLD = -50*10**-3#mV
MEMBRANE_RESISTANCE = 10.*10**6 #Mohm
MEMBRANE_TIME_SCALE = 8.*10**-3 #ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0*10**-3 #ms
u_rest=V_REST
tau=MEMBRANE_TIME_SCALE
R=MEMBRANE_RESISTANCE
I=2.1*10**-9
u=V_REST

derivative=(-(u-u_rest)+R*I)/tau
t_ts=range(0,1000)
u_ts=[]
for t in range(0,1000):
    u+=derivative*0.0001
    derivative=(-(u-u_rest)+R*I)/tau
    if u>=FIRING_THRESHOLD:
        print("Firing")
        u=V_RESET
    u_ts.append(u)
    #print(u)
plt.plot(t_ts,u_ts)
plt.show()