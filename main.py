# Importarea bibliotecilor necesare
import numpy as np  # Pentru calcule numerice
import matplotlib.pyplot as plt  # Pentru vizualizare
import imageio_ffmpeg  # Pentru exportul animației
import matplotlib as mpl  # Pentru configurări matplotlib
from matplotlib.animation import FuncAnimation, FFMpegWriter  # Pentru crearea animațiilor

# Configurarea căii către executabilul FFmpeg pentru exportul video
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# Definirea parametrilor inițiali
a, b = 5, 3  # Semi-axele elipsei (a = axa mare, b = axa mică)
alpha = 1.5 * np.pi  # Unghiul constant între razele OP și OQ
theta1 = np.pi / 6  # Unghiul inițial pentru raza OP
total_frames = 360  # Numărul total de cadre pentru animație
h = 2  # Înălțimea punctului H

# Crearea figurii și a axelor 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # Inițializarea axelor 3D

# Configurarea aspectului graficului 3D
ax.set_facecolor('white')  # Setarea culorii de fundal alb
ax.set_proj_type('persp')  # Utilizarea proiecției în perspectivă
ax.set_box_aspect([1, 1, 0.7])  # Ajustarea raportului de aspect al cutiei 3D

# Generarea punctelor pentru elipsă
t_vals = np.linspace(0, 2 * np.pi, 400)  # Valori unghiulare de la 0 la 2π
x_el = a * np.cos(t_vals)  # Coordonatele x ale elipsei
y_el = b * np.sin(t_vals)  # Coordonatele y ale elipsei

# Pregătirea și reprezentarea elipsei
ellipse_x = np.append(x_el, x_el[0])  # Închiderea elipsei adăugând primul punct la final
ellipse_y = np.append(y_el, y_el[0])
ellipse_z = np.zeros_like(ellipse_x) - 0.01  # Setarea coordonatelor z ușor sub planul z=0

# Reprezentarea elipsei cu linii gri transparente
ax.plot(ellipse_x, ellipse_y, ellipse_z, color='gray', linewidth=1, alpha=0.5)

# Colorarea elipsei folosind o hartă de culori
cmap_ellipse = plt.get_cmap('viridis')  # Harta de culori pentru segmentele elipsei
for i in range(len(t_vals) - 1):
    seg_x = x_el[i:i+2]  # Coordonatele x ale segmentului curent
    seg_y = y_el[i:i+2]  # Coordonatele y ale segmentului curent
    # Reprezentarea segmentului cu culoarea corespunzătoare poziției sale
    ax.plot(seg_x, seg_y, zs=0, color=cmap_ellipse(i / len(t_vals)), linewidth=2)

# Generarea și reprezentarea cercului
theta_vals = np.linspace(0, 2 * np.pi, 400)  # Valori unghiulare pentru cerc
x_circ = a * np.cos(theta_vals)  # Coordonatele x ale cercului cu raza a
y_circ = a * np.sin(theta_vals)  # Coordonatele y ale cercului

# Reprezentarea cercului cu linie punctată portocalie
circle = ax.plot(x_circ, y_circ, zs=0, zdir='z', linestyle='--', linewidth=1.5, color='darkorange', label='Cerc')
circle[0]._depthshade = True  # Activarea umbrelor de profunzime pentru cerc

# Reprezentarea originii sistemului de coordonate
O = np.array([0, 0])  # Coordonatele originii în planul xy
origin_point = ax.scatter(*O, 0, color='black', s=30)  # Punctul O reprezentat ca un punct negru
origin_point._depthshade = True  # Activarea umbrelor de profunzime pentru punctul O
ax.text(0, 0, 0, 'O', va='bottom', ha='right')  # Etichetarea punctului O

# Definirea punctelor cheie pentru vizualizare
P = a * np.array([np.cos(theta1), np.sin(theta1)])  # Punctul P pe cerc la unghiul theta1
Q = a * np.array([np.cos(theta1 + alpha), np.sin(theta1 + alpha)])  # Punctul Q pe cerc la unghiul theta1 + alpha

# Proiecțiile punctelor P și Q pe elipsă
D = np.array([P[0], np.sign(P[1]) * b * np.sqrt(1 - (P[0]**2) / a**2)])  # Proiecția lui P pe elipsă
E = np.array([Q[0], np.sign(Q[1]) * b * np.sqrt(1 - (Q[0]**2) / a**2)])  # Proiecția lui Q pe elipsă

# Punctul H ridicat la înălțimea h deasupra punctului D
H = np.array([D[0], D[1], h])  # Punctul H în spațiul 3D
# Reprezentarea razelor OP și OQ
op_line, = ax.plot([0, P[0]], [0, P[1]], zs=0, linestyle='--', linewidth=1.5, color='crimson', label='OP')  # Raza OP în roșu
oq_line, = ax.plot([0, Q[0]], [0, Q[1]], zs=0, linestyle='--', linewidth=1.5, color='purple', label='OQ')  # Raza OQ în violet

# Activarea umbrelor de profunzime pentru razele OP și OQ
op_line._depthshade = True
oq_line._depthshade = True

# Reprezentarea liniilor de construcție
PD_line, = ax.plot([P[0], D[0]], [P[1], D[1]], zs=[0, 0], linestyle=':', linewidth=1.2, color='gray')  # Linia PD (proiecția pe elipsă)
QE_line, = ax.plot([Q[0], E[0]], [Q[1], E[1]], zs=[0, 0], linestyle=':', linewidth=1.2, color='gray')  # Linia QE (proiecția pe elipsă)
DH_line, = ax.plot([D[0], D[0]], [D[1], D[1]], [0, h], linestyle=':', linewidth=1.2, color='gray')  # Linia DH (verticală)
HE_line, = ax.plot([H[0], E[0]], [H[1], E[1]], [H[2], 0], linestyle='-', linewidth=1.5, color='gray')  # Linia HE (diagonală)

# Activarea umbrelor de profunzime pentru toate liniile
PD_line._depthshade = True
QE_line._depthshade = True
DH_line._depthshade = True
HE_line._depthshade = True

# Reprezentarea punctelor P, Q, D, E și H
scatter_P = ax.scatter(P[0], P[1], 0, color='red', s=40)  # Punctul P în roșu
scatter_Q = ax.scatter(Q[0], Q[1], 0, color='blue', s=40)  # Punctul Q în albastru
scatter_D = ax.scatter(D[0], D[1], 0, color='green', s=40)  # Punctul D în verde
scatter_E = ax.scatter(E[0], E[1], 0, color='green', s=40)  # Punctul E în verde
scatter_H = ax.scatter(H[0], H[1], H[2], color='black', s=40)  # Punctul H în negru

# Activarea umbrelor de profunzime pentru toate punctele
scatter_P._depthshade = True
scatter_Q._depthshade = True
scatter_D._depthshade = True
scatter_E._depthshade = True
scatter_H._depthshade = True

# Adăugarea etichetelor pentru punctele P și Q
text_P = ax.text(P[0], P[1], 0, 'P', va='bottom', ha='left')  # Eticheta pentru punctul P
text_Q = ax.text(Q[0], Q[1], 0, 'Q', va='bottom', ha='left')  # Eticheta pentru punctul Q

# Configurarea legendei și aspectului cutiei 3D
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Poziționarea legendei în afara graficului
ax.set_box_aspect([1, 1, 0.5])  # Ajustarea raportului de aspect al cutiei 3D
ax.view_init(elev=30, azim=120)  # Setarea unghiului de vizualizare inițial
plt.tight_layout()  # Ajustarea layout-ului pentru a evita suprapunerile

# Definirea hărții de culori pentru traiectoria HE
eh_cmap = plt.get_cmap('plasma')  # Harta de culori pentru traiectoria HE

# Variabile globale pentru animație
trace_lines = []  # Lista pentru stocarea liniilor de traseu
base_zorder = 1000  # Valoarea de bază pentru ordinea de desenare (z-order)

# Funcția de actualizare pentru animație
def update(frame):
    global trace_lines  # Folosirea variabilei globale pentru liniile de traseu
    # Calcularea noilor poziții ale punctelor pentru cadrul curent
    th = theta1 + frame * np.pi/180  # Unghiul curent (creștere cu 1 grad per cadru)
    P = a * np.array([np.cos(th), np.sin(th)])  # Actualizarea poziției punctului P pe cerc
    Q = a * np.array([np.cos(th + alpha), np.sin(th + alpha)])  # Actualizarea poziției punctului Q pe cerc

    # Actualizarea proiecțiilor pe elipsă
    D = np.array([P[0], np.sign(P[1]) * b * np.sqrt(1 - (P[0]**2) / a**2)])  # Proiecția lui P pe elipsă
    E = np.array([Q[0], np.sign(Q[1]) * b * np.sqrt(1 - (Q[0]**2) / a**2)])  # Proiecția lui Q pe elipsă

    # Actualizarea poziției punctului H
    H = np.array([D[0], D[1], h])  # Punctul H deasupra lui D

    # Calculul distanțelor pentru efectele de profunzime
    dist_h = np.sqrt(H[0]**2 + H[1]**2 + H[2]**2)  # Distanța punctului H față de origine
    dist_e = np.sqrt(E[0]**2 + E[1]**2)  # Distanța punctului E față de origine

    # Selectarea culorii pentru linia HE curentă în funcție de progresul animației
    color = eh_cmap(frame / total_frames)  # Culoarea bazată pe cadrul curent

    # Calculul profunzimii medii pentru linia HE
    trace_depth = (dist_h + dist_e) / 2  # Profunzimea medie a liniei

    # Desenarea liniei HE curente și adăugarea ei la lista de traseu
    trace_line = ax.plot([H[0], E[0]], [H[1], E[1]], [H[2], 0], color=color, alpha=0.7, linewidth=1.5)
    trace_line[0].set_zorder(base_zorder - trace_depth)  # Setarea ordinii de desenare bazată pe profunzime
    trace_lines.append(trace_line[0])  # Adăugarea liniei la lista de traseu

    # Ajustarea grosimii și transparenței liniei în funcție de profunzime
    line_width = 1.5 + max(0, 1.0 - trace_depth/5)  # Liniile mai apropiate sunt mai groase
    line_alpha = min(0.8, max(0.4, 0.9 - trace_depth/10))  # Liniile mai apropiate sunt mai opace
    trace_line[0].set_linewidth(line_width)  # Aplicarea grosimii calculate
    trace_line[0].set_alpha(line_alpha)  # Aplicarea transparenței calculate

    # Ajustarea transparenței liniilor de traseu anterioare pentru a crea efectul de estompare
    for i, line in enumerate(trace_lines[:-1]):
        age_factor = (i + 1) / len(trace_lines)  # Factorul de vârstă al liniei (liniile mai vechi sunt mai transparente)
        if line in ax.lines:
            line.set_alpha(max(0.2, line_alpha * (1 - age_factor * 0.5)))  # Reducerea treptată a opacității

    # Limitarea numărului de linii de traseu pentru a evita încărcarea excesivă a memoriei
    max_traces = total_frames  # Numărul maxim de linii de traseu este egal cu numărul de cadre
    if len(trace_lines) > max_traces:
        oldest_line = trace_lines.pop(0)  # Eliminarea celei mai vechi linii
        if oldest_line in ax.lines:
            oldest_line.remove()  # Eliminarea liniei din grafic

    # Calculul direcției camerei pentru efectele de profunzime
    azim = np.radians(ax.azim)  # Convertirea azimutului în radiani
    elev = np.radians(ax.elev)  # Convertirea elevației în radiani

    # Vectorul direcției camerei calculat din unghiurile de azimut și elevație
    camera_dir = np.array([
        -np.cos(elev) * np.sin(azim),  # Componenta x
        -np.cos(elev) * np.cos(azim),  # Componenta y
        np.sin(elev)                   # Componenta z
    ])

    # Definirea punctelor în spațiul 3D pentru calculul distanțelor
    O_3d = np.array([0, 0, 0])           # Originea
    P_3d = np.array([P[0], P[1], 0])     # Punctul P
    Q_3d = np.array([Q[0], Q[1], 0])     # Punctul Q
    D_3d = np.array([D[0], D[1], 0])     # Punctul D
    E_3d = np.array([E[0], E[1], 0])     # Punctul E
    H_3d = np.array([H[0], H[1], H[2]])  # Punctul H

    # Calculul distanțelor proiectate pe direcția camerei (pentru efecte de profunzime)
    dist_o = np.dot(O_3d, camera_dir)  # Distanța originii
    dist_p = np.dot(P_3d, camera_dir)  # Distanța punctului P
    dist_q = np.dot(Q_3d, camera_dir)  # Distanța punctului Q
    dist_d = np.dot(D_3d, camera_dir)  # Distanța punctului D
    dist_e = np.dot(E_3d, camera_dir)  # Distanța punctului E
    dist_h = np.dot(H_3d, camera_dir)  # Distanța punctului H

    # Actualizarea poziției și proprietăților liniilor

    # Actualizarea liniei OP
    op_line.set_data([0, P[0]], [0, P[1]])  # Coordonate x și y
    op_line.set_3d_properties([0, 0])  # Coordonate z
    op_line.set_zorder(base_zorder - (dist_o + dist_p)/2)  # Actualizarea ordinii de desenare

    # Actualizarea liniei OQ
    oq_line.set_data([0, Q[0]], [0, Q[1]])
    oq_line.set_3d_properties([0, 0])
    oq_line.set_zorder(base_zorder - (dist_o + dist_q)/2)

    # Actualizarea liniei PD
    PD_line.set_data([P[0], D[0]], [P[1], D[1]])
    PD_line.set_3d_properties([0, 0])
    PD_line.set_zorder(base_zorder - (dist_p + dist_d)/2)

    # Actualizarea liniei QE
    QE_line.set_data([Q[0], E[0]], [Q[1], E[1]])
    QE_line.set_3d_properties([0, 0])
    QE_line.set_zorder(base_zorder - (dist_q + dist_e)/2)

    # Actualizarea liniei DH (verticală)
    DH_line.set_data([D[0], D[0]], [D[1], D[1]])
    DH_line.set_3d_properties([0, h])
    DH_line.set_zorder(base_zorder - (dist_d + dist_h)/2)

    # Actualizarea liniei HE
    HE_line.set_data([H[0], E[0]], [H[1], E[1]])
    HE_line.set_3d_properties([H[2], 0])
    HE_line.set_zorder(base_zorder - (dist_h + dist_e)/2)

    # Calcularea grosimii liniilor în funcție de distanța față de cameră
    # Liniile mai apropiate de cameră sunt afișate mai gros pentru a accentua efectul 3D
    op_width = 1.5 + max(0, 1.0 - (dist_o + dist_p)/10)  # Grosimea liniei OP
    oq_width = 1.5 + max(0, 1.0 - (dist_o + dist_q)/10)  # Grosimea liniei OQ
    pd_width = 1.2 + max(0, 0.8 - (dist_p + dist_d)/10)  # Grosimea liniei PD
    qe_width = 1.2 + max(0, 0.8 - (dist_q + dist_e)/10)  # Grosimea liniei QE
    dh_width = 1.2 + max(0, 0.8 - (dist_d + dist_h)/10)  # Grosimea liniei DH
    he_width = 1.5 + max(0, 1.0 - (dist_h + dist_e)/10)  # Grosimea liniei HE

    # Aplicarea grosimilor calculate la liniile respective
    op_line.set_linewidth(op_width)
    oq_line.set_linewidth(oq_width)
    PD_line.set_linewidth(pd_width)
    QE_line.set_linewidth(qe_width)
    DH_line.set_linewidth(dh_width)
    HE_line.set_linewidth(he_width)

    # Valoare de îmbunătățire a ordinii de desenare pentru puncte (pentru a fi deasupra liniilor)
    point_zorder_boost = 5

    # Actualizarea poziției și proprietăților punctului P
    scatter_P._offsets3d = ([P[0]], [P[1]], [0])  # Actualizarea coordonatelor 3D
    scatter_P.set_zorder(base_zorder - dist_p + point_zorder_boost)  # Actualizarea ordinii de desenare
    scatter_P.set_sizes([40 + max(0, 20 - dist_p*2)])  # Ajustarea dimensiunii în funcție de distanță

    # Actualizarea poziției și proprietăților punctului Q
    scatter_Q._offsets3d = ([Q[0]], [Q[1]], [0])
    scatter_Q.set_zorder(base_zorder - dist_q + point_zorder_boost)
    scatter_Q.set_sizes([40 + max(0, 20 - dist_q*2)])

    # Actualizarea poziției și proprietăților punctului D
    scatter_D._offsets3d = ([D[0]], [D[1]], [0])
    scatter_D.set_zorder(base_zorder - dist_d + point_zorder_boost)
    scatter_D.set_sizes([40 + max(0, 20 - dist_d*2)])

    # Actualizarea poziției și proprietăților punctului E
    scatter_E._offsets3d = ([E[0]], [E[1]], [0])
    scatter_E.set_zorder(base_zorder - dist_e + point_zorder_boost)
    scatter_E.set_sizes([40 + max(0, 20 - dist_e*2)])

    # Actualizarea poziției și proprietăților punctului H
    scatter_H._offsets3d = ([H[0]], [H[1]], [H[2]])
    scatter_H.set_zorder(base_zorder - dist_h + point_zorder_boost)
    scatter_H.set_sizes([40 + max(0, 20 - dist_h*2)])

    # Valoare de îmbunătățire a ordinii de desenare pentru etichete (pentru a fi deasupra punctelor)
    text_zorder_boost = 10

    # Actualizarea poziției și proprietăților etichetei P
    text_P.set_position((P[0], P[1]))  # Actualizarea poziției în planul xy
    text_P.set_3d_properties(0)  # Actualizarea coordonatei z
    text_P.set_zorder(base_zorder - dist_p + text_zorder_boost)  # Actualizarea ordinii de desenare
    text_P.set_alpha(min(1.0, max(0.5, 1.2 - dist_p/5)))  # Ajustarea transparenței în funcție de distanță

    # Actualizarea poziției și proprietăților etichetei Q
    text_Q.set_position((Q[0], Q[1]))
    text_Q.set_3d_properties(0)
    text_Q.set_zorder(base_zorder - dist_q + text_zorder_boost)
    text_Q.set_alpha(min(1.0, max(0.5, 1.2 - dist_q/5)))

                # Returnarea tuturor elementelor actualizate pentru animație
                # Aceasta include toate liniile, punctele, etichetele și liniile de traseu
    return [op_line, oq_line, PD_line, QE_line, DH_line, HE_line,
            scatter_P, scatter_Q, scatter_D, scatter_E, scatter_H, text_P, text_Q] + trace_lines

# Configurarea unghiului de vizualizare inițial
ax.view_init(elev=25, azim=45)  # Eleveție 25 grade, azimut 45 grade

# Setarea limitelor axelor pentru a încadra bine vizualizarea
ax.set_xlim(-a*1.2, a*1.2)  # Limitele axei x
ax.set_ylim(-a*1.2, a*1.2)  # Limitele axei y
ax.set_zlim(0, h*1.2)  # Limitele axei z (de la 0 la h*1.2)

# Configurarea aspectului panourilor de axe
ax.xaxis.pane.fill = False  # Panoul axei x transparent
ax.yaxis.pane.fill = False  # Panoul axei y transparent
ax.zaxis.pane.fill = False  # Panoul axei z transparent
ax.grid(True, linestyle='--', alpha=0.3)  # Activarea grilei cu linii punctate semi-transparente

# Adăugarea unui plan de bază semi-transparent
xx, yy = np.meshgrid(np.linspace(-a*1.2, a*1.2, 2), np.linspace(-a*1.2, a*1.2, 2))  # Crearea grilei pentru plan
zz = np.zeros_like(xx) - 0.01  # Poziționarea planului ușor sub z=0
ground = ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1, shade=True)  # Reprezentarea planului gri semi-transparent

# Crearea animației
ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)  # Funcția FuncAnimation apelează funcția update pentru fiecare cadru

# Activarea calculului ordinii de desenare pentru o afișare corectă a profunzimii
ax.computed_zorder = True

# Salvarea animației în format video sau GIF
try:
    # Încercarea de a salva animația ca fișier MP4 folosind FFmpeg
    writer = FFMpegWriter(fps=30)  # 30 de cadre pe secundă
    ani.save('angle_POQ_gradient.mp4', writer=writer)
    print("Animation saved as MP4.")
except Exception as e:
    # Dacă FFmpeg nu este disponibil, salvăm ca GIF folosind Pillow
    print(f"FFMpegWriter unavailable ({e}); saving as GIF.")
    ani.save('angle_POQ_gradient.gif', writer='pillow', fps=30)

# Afișarea animației
plt.show()