import matplotlib.patches as patches
import numpy as np

def draw_key(ax, key_x, key_y, scale_x, scale_y, color, zorder=1):
    cx = (key_x + 0.5) * scale_x
    cy = (key_y + 0.5) * scale_y

    s = min(scale_x, scale_y)

    head_outer = patches.Circle(
        (cx - 0.12 * s, cy),
        radius=0.22 * s,
        facecolor=color,
        edgecolor='none',
        zorder=zorder
    )
    ax.add_patch(head_outer)

    head_inner = patches.Circle(
        (cx - 0.12 * s, cy),
        radius=0.10 * s,
        facecolor='white',
        edgecolor='none',
        zorder=zorder + 0.1
    )
    ax.add_patch(head_inner)

    shaft = patches.Rectangle(
        (cx - 0.02 * s, cy - 0.05 * s),
        0.42 * s,
        0.10 * s,
        facecolor=color,
        edgecolor='none',
        zorder=zorder
    )
    ax.add_patch(shaft)

    tooth1 = patches.Rectangle(
        (cx + 0.22 * s, cy + 0.02 * s),
        0.07 * s,
        0.12 * s,
        facecolor=color,
        edgecolor='none',
        zorder=zorder
    )
    ax.add_patch(tooth1)

    tooth2 = patches.Rectangle(
        (cx + 0.33 * s, cy + 0.02 * s),
        0.07 * s,
        0.08 * s,
        facecolor=color,
        edgecolor='none',
        zorder=zorder
    )
    ax.add_patch(tooth2)
