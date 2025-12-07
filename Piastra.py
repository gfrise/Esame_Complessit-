"""
disegno_plate_v4.py
Versione: archi = porzioni DI CIRCONFERENZA con raggi crescente (man mano che si allontana dal centro).
- una famiglia sotto la diagonale (centro in basso-sinistra)
- una famiglia sopra la diagonale (centro in alto-destra)
- sfondo blu scuro, linee rosse
Dipendenze: numpy, matplotlib
Esegui: python disegno_plate_v4.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Parametri della piastra ----------
W, H = 5.3, 2.6            # proporzioni (larghezza, altezza)
FIG_DPI = 150
LINEWIDTH_BORDER = 6.0
LINEWIDTH_ARC_BASE = 3.5
N_ARCS_LEFT = 7            # famiglia sotto la diagonale (centro basso-sinistra)
N_ARCS_RIGHT = 7           # famiglia sopra la diagonale (centro alto-destra)
PAD = 0.06 * min(W, H)     # margine interno

# ---------- Colori ----------
PLATE_COLOR = "#071a52"    # blu scuro
LINE_COLOR = "#ff3b30"     # rosso acceso

# ---------- Setup figura ----------
fig = plt.figure(figsize=(W, H), dpi=FIG_DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_facecolor(PLATE_COLOR)

# ---------- Bordo ----------
rect = Rectangle((0, 0), W, H, linewidth=LINEWIDTH_BORDER, edgecolor=LINE_COLOR,
                 facecolor='none', joinstyle='round')
ax.add_patch(rect)

# ---------- Diagonale perfettamente dritta ----------
x_diag = np.linspace(0 + PAD*0.3, W - PAD*0.3, 1200)
y_diag = H - (x_diag / W) * H   # retta esatta
ax.plot(x_diag, y_diag, color=LINE_COLOR, linewidth=LINEWIDTH_BORDER * 0.95, solid_capstyle='round')

def y_on_diag(x):
    return H - (x / W) * H

# ---------- Famiglia LEFT: centro in basso-sinistra, archi SOLO sotto la diagonale ----------
cx_left, cy_left = PAD, PAD

# Definire raggi *crescente* man mano che ci si allontana dal centro:
# r_min è piccolo (vicino al centro), r_max grande (raggi che arrivano vicino alla periferia)
r_min_left = 0.12 * max(W, H)
r_max_left = np.hypot(W - cx_left, H - cy_left) * 0.98
# Raggi in ordine crescente: gli archi esterni hanno r più grande
radii_left = np.linspace(r_min_left, r_max_left, N_ARCS_LEFT)

# Angoli per porzione di circonferenza (solo il quarto interno ma esteso per la forma del disegno)
theta_left = np.linspace(-0.06, np.pi/2.05, 1200)

# Disegno: disegna dall'arco più piccolo a quello più grande (così i grandi stanno dietro)
for i, r in enumerate(radii_left):
    x = cx_left + r * np.cos(theta_left)
    y = cy_left + r * np.sin(theta_left)
    # taglio ai limiti del rettangolo
    mask_rect = (x >= 0) & (x <= W) & (y >= 0) & (y <= H)
    # solo sotto la diagonale
    mask_diag = y < y_on_diag(x)
    mask = mask_rect & mask_diag
    if np.any(mask):
        # linea leggermente più sottile per gli archi esterni (per profondità)
        lw = LINEWIDTH_ARC_BASE * (1.05 - 0.07 * (i / max(1, N_ARCS_LEFT-1)))
        ax.plot(x[mask], y[mask], color=LINE_COLOR, linewidth=lw, solid_capstyle='round')

# ---------- Famiglia RIGHT: centro in alto-destra, archi SOLO sopra la diagonale ----------
cx_right, cy_right = W - PAD, H - PAD

r_min_right = 0.12 * max(W, H)
r_max_right = np.hypot(cx_right, cy_right) * 0.98
# Raggi crescenti: piccoli vicino al centro, grandi verso l'esterno
radii_right = np.linspace(r_min_right, r_max_right, N_ARCS_RIGHT)

# angoli per il quarto in alto-destra (porzioni di circonferenza)
theta_right = np.linspace(np.pi - 0.06, 1.5 * np.pi + 0.06, 1200)

for i, r in enumerate(radii_right):
    x = cx_right + r * np.cos(theta_right)
    y = cy_right + r * np.sin(theta_right)
    mask_rect = (x >= 0) & (x <= W) & (y >= 0) & (y <= H)
    # solo sopra la diagonale
    mask_diag = y > y_on_diag(x)
    mask = mask_rect & mask_diag
    if np.any(mask):
        lw = LINEWIDTH_ARC_BASE * (1.05 - 0.07 * (i / max(1, N_ARCS_RIGHT-1)))
        ax.plot(x[mask], y[mask], color=LINE_COLOR, linewidth=lw, solid_capstyle='round')

# ---------- Dettaglio angolo in basso-sinistra (piccolo arrotondamento) ----------
theta_corner = np.linspace(np.pi/2, np.pi, 80)
r_corner = PAD * 0.9
cx_c, cy_c = PAD*0.45, PAD*0.45
x_c = cx_c + r_corner * np.cos(theta_corner)
y_c = cy_c + r_corner * np.sin(theta_corner)
mask_c = (x_c >= 0) & (x_c <= W) & (y_c >= 0) & (y_c <= H) & (y_c < y_on_diag(x_c))
ax.plot(x_c[mask_c], y_c[mask_c], color=LINE_COLOR, linewidth=LINEWIDTH_ARC_BASE * 1.05, solid_capstyle='round')

# ---------- Pulizia e salvataggio ----------
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_png = "piastra_archi_v4.png"
output_svg = "piastra_archi_v4.svg"
plt.savefig(output_png, dpi=FIG_DPI, facecolor=PLATE_COLOR, bbox_inches='tight', pad_inches=0)
plt.savefig(output_svg, dpi=FIG_DPI, facecolor=PLATE_COLOR, bbox_inches='tight', pad_inches=0)
print(f"Immagini salvate: '{output_png}' e '{output_svg}'")

plt.show()
