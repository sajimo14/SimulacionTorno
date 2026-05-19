"""
generar_graficas.py
-------------------
Genera las gráficas de la memoria del proyecto SimulacionTorno (Fase III/IV
de Arquitectura de los Computadores, Universidad de Alicante).

Salidas PNG en el directorio actual:
    - fig_tiempos_gpu.png
    - fig_speedup.png
    - fig_distribucion.png
    - fig_resumen_global.png

Autor: Grupo 7 (Samuel, Gabriel, Dairon, Niko, Bryan, Denis)
Repositorio: https://github.com/sajimo14/SimulacionTorno
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Paleta de colores estilo NVIDIA
# ---------------------------------------------------------------------------
NVIDIA_GREEN = "#76B900"
NVIDIA_GREEN_DARK = "#4f7a00"
ACCENT_CYAN = "#00d1ff"
ACCENT_AMBER = "#ffb000"
BG_DARK = "#0d0d0d"
BG_PANEL = "#181818"
GRID_COLOR = "#2a2a2a"
TEXT_COLOR = "#e0e0e0"
TEXT_DIM = "#9a9a9a"

# Color por rama
COLOR_MASTER = NVIDIA_GREEN          # rama final ganadora
COLOR_STRIDE = ACCENT_CYAN
COLOR_SHARED = ACCENT_AMBER

# Aplicar estilo base
plt.rcParams.update({
    "figure.facecolor":    BG_DARK,
    "axes.facecolor":      BG_PANEL,
    "axes.edgecolor":      GRID_COLOR,
    "axes.labelcolor":     TEXT_COLOR,
    "axes.titlecolor":     TEXT_COLOR,
    "xtick.color":         TEXT_DIM,
    "ytick.color":         TEXT_DIM,
    "text.color":          TEXT_COLOR,
    "grid.color":          GRID_COLOR,
    "grid.linestyle":      "--",
    "grid.linewidth":      0.5,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})

# ---------------------------------------------------------------------------
# Datos medidos
#
# Cada lista contiene los 5 tiempos de ejecución (en segundos) reportados
# por la propia función runTest del programa, recogidos del documento
# EJECUCIONES.pdf. Todas las mediciones se han tomado con la misma entrada
# (test.for) y los mismos argumentos.
# ---------------------------------------------------------------------------
RESULTADOS = {
    "Gabriel (RTX 4070)": {
        "cpu": [3.816216, 3.833444, 3.805362, 3.814935, 3.821075],
        "master": [0.073513, 0.063137, 0.062821, 0.061882, 0.062525],
        "shared": [0.104106, 0.095076, 0.095159, 0.094868, 0.094449],
        "stride": [0.100045, 0.095729, 0.095919, 0.093615, 0.093758],
    },
    "Niko (RTX 5060 Ti)": {
        "cpu": [3.079657, 2.986103, 3.272759, 3.016248, 3.098504],
        "master": [0.070597, 0.069501, 0.070061, 0.069386, 0.069521],
        "shared": [0.096008, 0.096598, 0.096143, 0.095591, 0.099191],
        "stride": [0.097133, 0.097412, 0.096717, 0.097531, 0.099127],
    },
    "Denis (RTX 3070 Ti)": {
        "cpu": [3.711821, 3.703505, 3.735114, 3.681595, 3.723466],
        "master": [0.085469, 0.085237, 0.084948, 0.085195, 0.095066],
        "shared": [0.154616, 0.129076, 0.128317, 0.129637, 0.127675],
        "stride": [0.129718, 0.126030, 0.126674, 0.126230, 0.127124],
    },
    "Dairon SB (RTX 4060)": {
        "cpu": [3.942727, 3.906629, 3.907387, 3.905976, 3.930168],
        "master": [0.115818, 0.115929, 0.115239, 0.115871, 0.115883],
        "shared": [0.184259, 0.185172, 0.206770, 0.185058, 0.185408],
        "stride": [0.207958, 0.185018, 0.185182, 0.179778, 0.185337],
    },
    "Bryan (GTX 1660 S)": {
        "cpu": [3.831837, 3.816007, 3.851356, 3.811768, 3.870395],
        "master": [0.204314, 0.208513, 0.193589, 0.200728, 0.207499],
        "shared": [0.378124, 0.254071, 0.248616, 0.257595, 0.274887],
        "stride": [0.252472, 0.256229, 0.244812, 0.270669, 0.263136],
    },
    "Dairon PT (GTX 1650)": {
        "cpu": [5.880806, 5.987486, 6.021495, 6.051434, 6.177973],
        "master": [0.258024, 0.260835, 0.278336, 0.276341, 0.256018],
        "shared": [0.352984, 0.350528, 0.350783, 0.350645, 0.352218],
        "stride": [0.374438, 0.350861, 0.351847, 0.351177, 0.351371],
    },
}

EQUIPOS = list(RESULTADOS.keys())
RAMAS = [("master", "Master (final)", COLOR_MASTER),
         ("stride", "Grid-stride",    COLOR_STRIDE),
         ("shared", "Shared memory",  COLOR_SHARED)]


# ---------------------------------------------------------------------------
# Barrido del tamaño de bloque (rama master, RTX 4070)
#
# Speedup observado (tiempo_CPU / tiempo_GPU) para distintos tamaños de
# bloque, manteniendo todo lo demás constante. 5 ejecuciones por punto.
# ---------------------------------------------------------------------------
BARRIDO_BLOCK = {
    16:   [30.84, 30.72, 31.17, 31.11, 30.94],
    32:   [63.56, 63.02, 59.35, 62.11, 62.07],
    64:   [55.58, 57.65, 56.64, 57.34, 56.31],
    128:  [56.86, 56.13, 56.46, 56.85, 55.10],
    256:  [42.47, 42.52, 41.88, 42.29, 43.05],
    512:  [43.55, 41.81, 42.75, 42.73, 42.51],
    1024: [21.57, 21.45, 24.09, 21.42, 21.61],
}


def media(lista):
    return mean(lista)


def desv(lista):
    return stdev(lista) if len(lista) > 1 else 0.0


# ---------------------------------------------------------------------------
# Helpers de estilo
# ---------------------------------------------------------------------------
def title_panel(fig, title, subtitle=None):
    height_in = fig.get_size_inches()[1]
    title_y = 1.0 - 0.45 / height_in
    sub_y   = title_y - 0.40 / height_in
    fig.suptitle(title, color=NVIDIA_GREEN, fontsize=14, fontweight="bold",
                 x=0.04, ha="left", y=title_y)
    if subtitle:
        fig.text(0.04, sub_y, subtitle, color=TEXT_DIM, fontsize=9, ha="left")


def annotate_value(ax, x, y, txt, color=TEXT_COLOR, dy=0.003):
    ax.text(x, y + dy, txt, ha="center", va="bottom",
            color=color, fontsize=7.5, fontweight="bold")


# ---------------------------------------------------------------------------
# Figura 1: tiempos GPU medios por equipo y rama (barras agrupadas)
# ---------------------------------------------------------------------------
def fig_tiempos_gpu():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    title_panel(fig,
                "Tiempo medio de ejecución en GPU",
                "Media de 5 ejecuciones por configuración · valores en segundos · menor es mejor")

    x = np.arange(len(EQUIPOS))
    width = 0.26

    for i, (key, label, color) in enumerate(RAMAS):
        medias = [media(RESULTADOS[eq][key]) for eq in EQUIPOS]
        sds    = [desv(RESULTADOS[eq][key]) for eq in EQUIPOS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, medias, width, yerr=sds,
                      color=color, edgecolor=BG_DARK, linewidth=0.8,
                      label=label, capsize=3,
                      error_kw=dict(ecolor=TEXT_DIM, lw=0.8))
        for bar, m in zip(bars, medias):
            annotate_value(ax, bar.get_x() + bar.get_width() / 2,
                           bar.get_height(), f"{m*1000:.0f} ms",
                           color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(EQUIPOS, fontsize=9)
    ax.set_ylabel("Tiempo GPU (s)")
    leg = ax.legend(facecolor=BG_PANEL, edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, loc="upper left", framealpha=0.9)
    plt.setp(leg.get_texts(), color=TEXT_COLOR)
    ax.set_ylim(0, max(max(media(RESULTADOS[eq][k]) for eq in EQUIPOS)
                       for k in ["master", "stride", "shared"]) * 1.18)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig("fig_tiempos_gpu.png", dpi=160, facecolor=BG_DARK)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 2: speedups por equipo y rama
# ---------------------------------------------------------------------------
def fig_speedup():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    title_panel(fig,
                "Factor de aceleración (speedup) respecto a CPU",
                "speedup = tiempo_CPU / tiempo_GPU · mayor es mejor")

    x = np.arange(len(EQUIPOS))
    width = 0.26

    for i, (key, label, color) in enumerate(RAMAS):
        speedups = [media(RESULTADOS[eq]["cpu"]) / media(RESULTADOS[eq][key])
                    for eq in EQUIPOS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, speedups, width,
                      color=color, edgecolor=BG_DARK, linewidth=0.8,
                      label=label)
        for bar, sp in zip(bars, speedups):
            annotate_value(ax, bar.get_x() + bar.get_width() / 2,
                           bar.get_height(), f"{sp:.1f}×",
                           color=color, dy=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(EQUIPOS, fontsize=9)
    ax.set_ylabel("Speedup (×)")
    leg = ax.legend(facecolor=BG_PANEL, edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, loc="upper right", framealpha=0.9)
    plt.setp(leg.get_texts(), color=TEXT_COLOR)
    ax.set_ylim(0, max([media(RESULTADOS[eq]["cpu"]) /
                        media(RESULTADOS[eq]["master"])
                        for eq in EQUIPOS]) * 1.18)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig("fig_speedup.png", dpi=160, facecolor=BG_DARK)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 3: panel resumen — speedup medio por rama (todos los equipos)
# ---------------------------------------------------------------------------
def fig_resumen_global():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    title_panel(fig,
                "Speedup medio global por configuración",
                "Promedio entre los 6 equipos del grupo")

    speedups_por_rama = {}
    for key, label, color in RAMAS:
        speedups = [media(RESULTADOS[eq]["cpu"]) / media(RESULTADOS[eq][key])
                    for eq in EQUIPOS]
        speedups_por_rama[key] = {
            "label": label,
            "color": color,
            "media": mean(speedups),
            "min":   min(speedups),
            "max":   max(speedups),
        }

    keys = list(speedups_por_rama.keys())
    y = np.arange(len(keys))
    medias = [speedups_por_rama[k]["media"] for k in keys]
    mins   = [speedups_por_rama[k]["min"]   for k in keys]
    maxs   = [speedups_por_rama[k]["max"]   for k in keys]
    colors = [speedups_por_rama[k]["color"] for k in keys]
    labels = [speedups_por_rama[k]["label"] for k in keys]

    for yi, m, lo, hi, c in zip(y, medias, mins, maxs, colors):
        ax.barh(yi, m, color=c, edgecolor=BG_DARK, linewidth=0.8, height=0.55)
        ax.plot([lo, hi], [yi, yi], color=TEXT_DIM, linewidth=2, zorder=3)
        ax.scatter([lo, hi], [yi, yi], color=TEXT_DIM, s=18, zorder=4)
        ax.text(m + 1, yi, f"  media {m:.1f}×   (rango {lo:.1f}×–{hi:.1f}×)",
                va="center", color=TEXT_COLOR, fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Speedup (×)")
    ax.set_xlim(0, max(maxs) * 1.4)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig("fig_resumen_global.png", dpi=160, facecolor=BG_DARK)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 4: barrido del tamaño de bloque
# ---------------------------------------------------------------------------
def fig_block_sweep():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    title_panel(fig,
                "Barrido del tamaño de bloque (rama master)",
                "RTX 4070 · 5 ejecuciones por configuración · "
                "speedup respecto a CPU · mayor es mejor")

    sizes  = list(BARRIDO_BLOCK.keys())
    medias = [mean(BARRIDO_BLOCK[s]) for s in sizes]
    sds    = [desv(BARRIDO_BLOCK[s]) for s in sizes]

    x = np.arange(len(sizes))

    best_idx = int(np.argmax(medias))
    colors = [NVIDIA_GREEN if i == best_idx else "#3a3a3a" for i in range(len(sizes))]
    edgecolors = [NVIDIA_GREEN if i == best_idx else GRID_COLOR for i in range(len(sizes))]

    bars = ax.bar(x, medias, width=0.62, color=colors,
                  edgecolor=edgecolors, linewidth=1.2,
                  yerr=sds, capsize=4,
                  error_kw=dict(ecolor=TEXT_DIM, lw=0.9))

    for i, (bar, m) in enumerate(zip(bars, medias)):
        col = NVIDIA_GREEN if i == best_idx else TEXT_COLOR
        weight = "bold" if i == best_idx else "normal"
        ax.text(bar.get_x() + bar.get_width() / 2, m + 1.2,
                f"{m:.1f}×", ha="center", va="bottom",
                color=col, fontsize=10, fontweight=weight)

    ax.plot(x, medias, color=NVIDIA_GREEN, alpha=0.35,
            linewidth=1.2, linestyle="--", zorder=1)

    labels = [f"{s}\n({s//32 if s>=32 else '½'} warp{'s' if s>32 else ''})"
              for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel("threads por bloque", labelpad=8)
    ax.set_ylabel("Speedup (×)")
    ax.set_ylim(0, max(medias) * 1.18)

    ax.annotate("óptimo\n(1 warp exacto)",
                xy=(best_idx, medias[best_idx]),
                xytext=(best_idx + 1.4, medias[best_idx] + 6),
                color=NVIDIA_GREEN, fontsize=9.5,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=NVIDIA_GREEN,
                                lw=1.0, alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig("fig_block_sweep.png", dpi=160, facecolor=BG_DARK)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabla por consola (sirve para verificar los números que llevamos al texto)
# ---------------------------------------------------------------------------
def tabla_resumen():
    print("\n" + "=" * 96)
    print(f"{'Equipo':<24}{'CPU (s)':>10}{'master':>11}{'stride':>11}"
          f"{'shared':>11}{'SU master':>13}{'SU stride':>13}{'SU shared':>13}")
    print("-" * 96)
    for eq in EQUIPOS:
        cpu = media(RESULTADOS[eq]["cpu"])
        m   = media(RESULTADOS[eq]["master"])
        s   = media(RESULTADOS[eq]["stride"])
        sh  = media(RESULTADOS[eq]["shared"])
        print(f"{eq:<24}{cpu:>10.3f}{m:>11.4f}{s:>11.4f}{sh:>11.4f}"
              f"{cpu/m:>12.1f}x{cpu/s:>12.1f}x{cpu/sh:>12.1f}x")
    print("=" * 96)


if __name__ == "__main__":
    np.random.seed(42)
    fig_tiempos_gpu()
    fig_speedup()
    fig_resumen_global()
    fig_block_sweep()
    tabla_resumen()
    print("\nFiguras generadas:")
    print("  fig_tiempos_gpu.png")
    print("  fig_speedup.png")
    print("  fig_resumen_global.png")
    print("  fig_block_sweep.png")