#!/usr/bin/env python
#coding=utf-8
r"""Figure. Distribution of the total membrane current density in
different sections of a
neuron. Neurites of reconstructed neuron (center) are colored
according to the peak-to-peak
amplitude of membrane current intensity involved in generation of
action potential (colorbar in the bottom right corner). For clarity
axon was truncated. Two sections of the neuron are shown in
magnification: area surrounding cell's soma (\textbf{{a}}) and a piece of axon with
visible single node of Ranvier (\textbf{{b}}, red color corresponds to high current
intensity typical for generation of action potential in a node of
Ranvier) """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, patches

from eap import field, graph
import os

import platform
ARCH = platform.machine()

def plot_contour(x_range, y_range, n_contours=15):
    xx, yy = field.calc_grid(x_range, y_range, n_samp=Nsamp)
    vext_p2p = field.estimate_on_grid(coords, I_p2p, xx, yy)
    graph.logcontour(xx, yy, vext_p2p[0, :], n_contours=n_contours, linecolors='0.8', linewidths=0.2, unit='nV')

dt = 0.025
tstop = 50

# Parameters
rho    = 3.5  #conductivity, Ohm.m
cutoff = 800. #high-pass cutoff, Hz
order  = 401  #filter order
Nsamp  = 30
filter = None
y_range = (-200, 610)
x_range = (-550, 250)
bg_color = (0.7, 0.7, 0.7)

simulation_filename = "neuron_simulation_data.npz"

# Simulation
if not os.path.exists(simulation_filename):
    from eap import cell
    cell.load_model('models/Mainen/demo_ext.hoc',
                    'models/Mainen/{}/.libs/libnrnmech.so'.format(ARCH))
    cell.initialize(dt=dt)
    t, I = cell.integrate(tstop)
    coords = cell.get_seg_coords()
    I_p2p = I.max(0) - I.min(0)
    I_p2p = I_p2p[None, :]
    np.savez(simulation_filename, coords=coords, I_p2p=I_p2p, t=t)
else:
    data = np.load(simulation_filename)
    coords = data['coords']
    t = data['t']
    I_p2p = data['I_p2p']

xx, yy = field.calc_grid(x_range, y_range, n_samp=Nsamp)
v_ext = field.estimate_on_grid(coords, I_p2p, xx, yy)


# Plots
fig = plt.figure(figsize=(6,6), facecolor=bg_color)
fig.subplots_adjust(left=0.05, right=0.95)
ax = plt.subplot(111, frameon=False)

#contour
plot_contour(x_range, y_range, 5)


#neuron
S = np.pi*coords['diam']*coords['L'] #segment surface
p2p = I_p2p[0, :]
norm = colors.LogNorm(vmin=p2p.min(), vmax=p2p.max())
col = graph.plot_neuron(coords, p2p, norm=norm, show_diams=True, width_min=1., width_max=6)
plt.xticks([])
plt.yticks([])
plt.xlim(x_range)
plt.ylim(y_range)



# scalebar
xp, yp = -500, -100
w, h = 100, 100
plt.plot([xp, xp], [yp, yp+h], 'k-')
plt.plot([xp, xp+h], [yp, yp], 'k-')
plt.text(xp-10, yp+h/2., u"100 µm", ha='right', va='center',
         transform=ax.transData)
plt.text(xp+h/2., yp-10, u"100 µm", ha='center', va='top',
         transform=ax.transData)

# color bar
ax_cbar=plt.axes([0.8, 0.15, 0.02, 0.3], frameon=False)
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
cbar = plt.colorbar(col,format=fmt, cax=ax_cbar)
cbar.ax.set_ylabel("current intensity\n($\mathrm{\\mu A/cm^2}$)", ma='center')
cbar.outline.set_visible(False)


# zoom soma area
ax_zoom1 = plt.axes([0.15, 0.5, 0.2, 0.2], axisbg=bg_color)
graph.plot_neuron(coords, p2p, norm=norm, show_diams=True, width_min=1.5)
plt.axis('scaled')
x1, x2 = (-180, -130)
y1, y2 = (-50, 0)
plot_contour((x1, x2), (y1, y2), 5)
plt.xlim((x1, x2))
plt.ylim((y1, y2))
plt.xticks([])
plt.yticks([])
plt.text(0.05, 0.85, 'a', transform=ax_zoom1.transAxes, weight='bold')

rec_zoom1 = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              transform=ax.transData,
                             fill=False,
                             ls='dotted')
fig.patches.append(rec_zoom1)

# zoom axon area
ax_zoom2 = plt.axes([0.65, 0.5, 0.2, 0.2], axisbg=bg_color)

x1, x2 = (-180, -130)
y1, y2 = (-165, -115)

plot_contour((x1, x2), (y1, y2), 5)
graph.plot_neuron(coords, p2p, norm=norm, show_diams=True, width_min=3)
plt.axis('scaled')
plt.xlim((x1, x2))
plt.ylim((y1, y2))
plt.xticks([])
plt.yticks([])
plt.text(0.05, 0.85, 'b', transform=ax_zoom2.transAxes, weight='bold')
rec_zoom2 = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              transform=ax.transData,
                             fill=False,
                             ls='dotted')
fig.patches.append(rec_zoom2)

plt.savefig('hunter.pdf', facecolor=fig.get_facecolor())
