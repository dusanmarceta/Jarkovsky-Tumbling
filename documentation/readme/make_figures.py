'''Figures for README.md (light and dark versions). Run from the repository root:
    python documentation/readme/make_figures.py
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  series=['#2a78d6', '#eb6834', '#1baf7a']),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 series=['#3987e5', '#d95926', '#199e70']),
}


def style(ax, t):
    ax.set_facecolor(t['surface'])
    ax.grid(True, color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['axis'])
    ax.tick_params(colors=t['muted'], labelcolor=t['ink2'], labelsize=9)
    ax.xaxis.label.set_color(t['ink2'])
    ax.yaxis.label.set_color(t['ink2'])
    ax.title.set_color(t['ink'])


def legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for text in leg.get_texts():
        text.set_color(t['ink2'])


# Drift per orbit, 80-facet sphere, TI 2500, a = 1 au, obliquity 90 deg, no orbital_initialisation (2026-10-14)
NO_CORRECTION = [-7.320842, -7.938039, -7.833575, -7.763023, -7.715801, -7.684091, -7.662696]
WITH_CORRECTION = [-7.320842, -7.766582, -7.613479, -7.618828]
LIMIT = -7.619270   # Aitken extrapolation of the run without correction (orbits 5-7)


def convergence(t, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(t['surface'])
    style(ax, t)
    ax.axhline(LIMIT, color=t['muted'], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(7.15, LIMIT + 0.012, 'converged drift', color=t['ink2'], fontsize=9, ha='right', va='bottom')
    for values, color, label in [(NO_CORRECTION, t['series'][1], 'without deep-temperature correction'),
                                 (WITH_CORRECTION, t['series'][0], 'with deep-temperature correction')]:
        x = np.arange(1, len(values) + 1)
        ax.plot(x, values, color=color, linewidth=2, marker='o', markersize=6,
                markeredgecolor=t['surface'], markeredgewidth=1.5, label=label, zorder=3)
    ax.annotate('stopped: change < 1e-3', xy=(4, WITH_CORRECTION[-1]), xytext=(4.25, -7.50),
                color=t['ink2'], fontsize=9, arrowprops=dict(arrowstyle='-', color=t['muted'], linewidth=0.8))
    ax.set_xlabel('orbit')
    ax.set_ylabel('mean drift over the orbit  [10⁻⁶ m/s]')
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.7, 7.3)
    ax.set_ylim(-8.0, -7.25)
    legend(ax, t, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, facecolor=t['surface'])
    plt.close(fig)


def read_comparison():
    rows = []
    for line in open(os.path.join(REPO, 'results/linear_comparison/results_2026-10-14.txt')):
        parts = line.split('|')
        if len(parts) == 4 and parts[0].split() and parts[0].split()[0].isdigit():
            g, P, TI = (int(v) for v in parts[0].split())
            lin, full, linT = (float(v) for v in parts[1].split())
            rows.append((g, P, TI, lin, full, linT))
    return np.array(rows)


def validation(t, path):
    data = read_comparison()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=150)
    fig.patch.set_facecolor(t['surface'])
    panels = [(4, 'Full model (T⁴ emission) vs linear theory'), (5, 'Linearized emission vs linear theory')]
    for ax, (column, title) in zip(axes, panels):
        style(ax, t)
        ax.axhline(0, color=t['axis'], linewidth=1)
        if column == 5:
            ax.axhline(1.6, color=t['muted'], linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(2600, 1.6, '80-facet\nmesh bias', color=t['ink2'], fontsize=8, va='center', ha='left')
        for gamma, color in zip((0, 45, 90), t['series']):
            sel = (data[:, 0] == gamma) & (data[:, 1] == 12)
            TI = data[sel, 2]
            rel = 100 * (data[sel, column] / data[sel, 3] - 1)
            ax.plot(TI, rel, color=color, linewidth=2, marker='o', markersize=6,
                    markeredgecolor=t['surface'], markeredgewidth=1.5, label=f'obliquity {gamma}°', zorder=3)
            # direct labels; 0° and 45° nearly coincide in the linearized panel, so they share one label
            if column == 5 and gamma == 45:
                continue
            label = '0°, 45°' if column == 5 and gamma == 0 else f'{gamma}°'
            va = {0: 'bottom', 45: 'top', 90: 'center'}[gamma] if column == 4 else ('bottom' if gamma == 0 else 'center')
            ax.text(TI[0] / 1.12, rel[0], label, color=t['ink2'], fontsize=9, ha='right', va=va)
        ax.set_xscale('log')
        ax.set_xticks([200, 500, 2000])
        ax.set_xticklabels(['200', '500', '2000'])
        ax.minorticks_off()
        ax.set_xlim(100, 4500)
        ax.set_xlabel('thermal inertia  [J m⁻² K⁻¹ s⁻½]')
        ax.set_ylabel('drift difference  [%]')
        ax.set_title(title, fontsize=10, loc='left', color=t['ink'])
    axes[1].set_ylim(-1, 5)
    legend(axes[0], t, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, facecolor=t['surface'])
    plt.close(fig)


if __name__ == '__main__':
    for name, t in THEMES.items():
        convergence(t, os.path.join(HERE, f'convergence_{name}.png'))
        validation(t, os.path.join(HERE, f'validation_{name}.png'))
