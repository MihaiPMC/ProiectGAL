import numpy as np
import matplotlib.pyplot as plt
import imageio_ffmpeg
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Set ffmpeg path
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# Ellipse parameters
a, b = 5, 3  # semiaxe

# Angle parameters
alpha = 1.5 * np.pi  # constant angle between OP and OQ
theta1 = np.pi / 6    # initial orientation (30°)
total_frames = 360

# Height for H
h = 2

# Prepare figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Configure axis for better depth perception
ax.set_facecolor('white')  # Clean background
ax.set_proj_type('persp')  # Use perspective projection for better 3D effect
ax.set_box_aspect([1, 1, 0.7])  # Slightly taller z-axis for better depth

# Static ellipse in gradient along its perimeter
t_vals = np.linspace(0, 2 * np.pi, 400)
x_el = a * np.cos(t_vals)
y_el = b * np.sin(t_vals)

# Create a polygon for the ellipse with a slight elevation for better depth perception
ellipse_x = np.append(x_el, x_el[0])  # Close the loop
ellipse_y = np.append(y_el, y_el[0])
ellipse_z = np.zeros_like(ellipse_x) - 0.01  # Slightly below z=0 plane for consistent depth ordering
ax.plot(ellipse_x, ellipse_y, ellipse_z, color='gray', linewidth=1, alpha=0.5)

# Use a colormap for the ellipse
cmap_ellipse = plt.get_cmap('viridis')
for i in range(len(t_vals) - 1):
    seg_x = x_el[i:i+2]
    seg_y = y_el[i:i+2]
    ax.plot(seg_x, seg_y, zs=0, color=cmap_ellipse(i / len(t_vals)), linewidth=2)

# Static circle along major axis diameter
theta_vals = np.linspace(0, 2 * np.pi, 400)
x_circ = a * np.cos(theta_vals)
y_circ = a * np.sin(theta_vals)
circle = ax.plot(x_circ, y_circ, zs=0, zdir='z', linestyle='--', linewidth=1.5, color='darkorange', label='Cerc')
circle[0]._depthshade = True

# Origin
O = np.array([0, 0])
origin_point = ax.scatter(*O, 0, color='black', s=30)
origin_point._depthshade = True
ax.text(0, 0, 0, 'O', va='bottom', ha='right')

# Prepare dynamic lines and points (initialized at theta1)
P = a * np.array([np.cos(theta1), np.sin(theta1)])
Q = a * np.array([np.cos(theta1 + alpha), np.sin(theta1 + alpha)])
D = np.array([P[0], np.sign(P[1]) * b * np.sqrt(1 - (P[0]**2) / a**2)])
E = np.array([Q[0], np.sign(Q[1]) * b * np.sqrt(1 - (Q[0]**2) / a**2)])
H = np.array([D[0], D[1], h])

# OP and OQ lines
op_line, = ax.plot([0, P[0]], [0, P[1]], zs=0, linestyle='--', linewidth=1.5, color='crimson', label='OP')
oq_line, = ax.plot([0, Q[0]], [0, Q[1]], zs=0, linestyle='--', linewidth=1.5, color='purple', label='OQ')
op_line._depthshade = True
oq_line._depthshade = True

# PD, QE, DH, HE
PD_line, = ax.plot([P[0], D[0]], [P[1], D[1]], zs=[0, 0], linestyle=':', linewidth=1.2, color='gray')
QE_line, = ax.plot([Q[0], E[0]], [Q[1], E[1]], zs=[0, 0], linestyle=':', linewidth=1.2, color='gray')
DH_line, = ax.plot([D[0], D[0]], [D[1], D[1]], [0, h], linestyle=':', linewidth=1.2, color='gray')
HE_line, = ax.plot([H[0], E[0]], [H[1], E[1]], [H[2], 0], linestyle='-', linewidth=1.5, color='gray')
PD_line._depthshade = True
QE_line._depthshade = True
DH_line._depthshade = True
HE_line._depthshade = True

# Scatter points P, Q, D, E, H
scatter_P = ax.scatter(P[0], P[1], 0, color='red', s=40)
scatter_Q = ax.scatter(Q[0], Q[1], 0, color='blue', s=40)
scatter_D = ax.scatter(D[0], D[1], 0, color='green', s=40)
scatter_E = ax.scatter(E[0], E[1], 0, color='green', s=40)
scatter_H = ax.scatter(H[0], H[1], H[2], color='black', s=40)
scatter_P._depthshade = True
scatter_Q._depthshade = True
scatter_D._depthshade = True
scatter_E._depthshade = True
scatter_H._depthshade = True
text_P = ax.text(P[0], P[1], 0, 'P', va='bottom', ha='left')
text_Q = ax.text(Q[0], Q[1], 0, 'Q', va='bottom', ha='left')

# Legend and view adjustment
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax.set_box_aspect([1, 1, 0.5])
ax.view_init(elev=30, azim=120)
plt.tight_layout()

# Colormap for EH trace
eh_cmap = plt.get_cmap('plasma')

# Global variables for animation
trace_lines = []
base_zorder = 1000  # Base z-order value for depth sorting

# Animation update function
def update(frame):
    global trace_lines
    th = theta1 + frame * np.pi/180
    # Recompute points
    P = a * np.array([np.cos(th), np.sin(th)])
    Q = a * np.array([np.cos(th + alpha), np.sin(th + alpha)])
    D = np.array([P[0], np.sign(P[1]) * b * np.sqrt(1 - (P[0]**2) / a**2)])
    E = np.array([Q[0], np.sign(Q[1]) * b * np.sqrt(1 - (Q[0]**2) / a**2)])
    H = np.array([D[0], D[1], h])

    # Calculate distance for depth ordering
    dist_h = np.sqrt(H[0]**2 + H[1]**2 + H[2]**2)
    dist_e = np.sqrt(E[0]**2 + E[1]**2)

    # Draw a colored trace EH with gradient
    color = eh_cmap(frame / total_frames)

    # Calculate trace line depth (average of endpoints)
    trace_depth = (dist_h + dist_e) / 2

    # Create new trace line with proper depth
    trace_line = ax.plot([H[0], E[0]], [H[1], E[1]], [H[2], 0], color=color, alpha=0.7, linewidth=1.5)
    # Use global base_zorder for consistent depth ordering
    trace_line[0].set_zorder(base_zorder - trace_depth)
    trace_lines.append(trace_line[0])

    # Adjust width and alpha of trace lines based on distance
    line_width = 1.5 + max(0, 1.0 - trace_depth/5)
    line_alpha = min(0.8, max(0.4, 0.9 - trace_depth/10))
    trace_line[0].set_linewidth(line_width)
    trace_line[0].set_alpha(line_alpha)

    # Ensure all trace lines remain visible by refreshing their appearance
    for i, line in enumerate(trace_lines[:-1]):  # Skip the newest line we just added
        # Gradually reduce alpha for older lines to create fading effect
        age_factor = (i + 1) / len(trace_lines)
        if line in ax.lines:
            line.set_alpha(max(0.2, line_alpha * (1 - age_factor * 0.5)))

    # Maintain trace lines throughout the full animation
    max_traces = total_frames  # Keep traces for the full rotation
    if len(trace_lines) > max_traces:
        oldest_line = trace_lines.pop(0)
        if oldest_line in ax.lines:
            oldest_line.remove()

    # Get current camera position (azimuth and elevation)
    azim = np.radians(ax.azim)
    elev = np.radians(ax.elev)

    # Create a camera-to-point direction vector based on current view
    camera_dir = np.array([
        -np.cos(elev) * np.sin(azim),
        -np.cos(elev) * np.cos(azim),
        np.sin(elev)
    ])

    # Calculate camera-based distances (projections onto camera direction)
    # For 2D points, extend them to 3D with z=0
    O_3d = np.array([0, 0, 0])
    P_3d = np.array([P[0], P[1], 0])
    Q_3d = np.array([Q[0], Q[1], 0])
    D_3d = np.array([D[0], D[1], 0])
    E_3d = np.array([E[0], E[1], 0])
    H_3d = np.array([H[0], H[1], H[2]])

    # Calculate dot products with camera direction for depth ordering
    # Larger dot product = farther from camera
    dist_o = np.dot(O_3d, camera_dir)
    dist_p = np.dot(P_3d, camera_dir)
    dist_q = np.dot(Q_3d, camera_dir)
    dist_d = np.dot(D_3d, camera_dir)
    dist_e = np.dot(E_3d, camera_dir)
    dist_h = np.dot(H_3d, camera_dir)

    # Calculate line depths as average of endpoint depths
    # Objects with higher zorder value appear in front
    # Using the global base_zorder value

    # Update dynamic lines with proper depth ordering
    op_line.set_data([0, P[0]], [0, P[1]])
    op_line.set_3d_properties([0, 0])
    op_line.set_zorder(base_zorder - (dist_o + dist_p)/2)

    oq_line.set_data([0, Q[0]], [0, Q[1]])
    oq_line.set_3d_properties([0, 0])
    oq_line.set_zorder(base_zorder - (dist_o + dist_q)/2)

    PD_line.set_data([P[0], D[0]], [P[1], D[1]])
    PD_line.set_3d_properties([0, 0])
    PD_line.set_zorder(base_zorder - (dist_p + dist_d)/2)

    QE_line.set_data([Q[0], E[0]], [Q[1], E[1]])
    QE_line.set_3d_properties([0, 0])
    QE_line.set_zorder(base_zorder - (dist_q + dist_e)/2)

    DH_line.set_data([D[0], D[0]], [D[1], D[1]])
    DH_line.set_3d_properties([0, h])
    DH_line.set_zorder(base_zorder - (dist_d + dist_h)/2)

    HE_line.set_data([H[0], E[0]], [H[1], E[1]])
    HE_line.set_3d_properties([H[2], 0])
    HE_line.set_zorder(base_zorder - (dist_h + dist_e)/2)

    # Also adjust line width based on distance (closer = thicker)
    op_width = 1.5 + max(0, 1.0 - (dist_o + dist_p)/10)
    oq_width = 1.5 + max(0, 1.0 - (dist_o + dist_q)/10)
    pd_width = 1.2 + max(0, 0.8 - (dist_p + dist_d)/10)
    qe_width = 1.2 + max(0, 0.8 - (dist_q + dist_e)/10)
    dh_width = 1.2 + max(0, 0.8 - (dist_d + dist_h)/10)
    he_width = 1.5 + max(0, 1.0 - (dist_h + dist_e)/10)

    op_line.set_linewidth(op_width)
    oq_line.set_linewidth(oq_width)
    PD_line.set_linewidth(pd_width)
    QE_line.set_linewidth(qe_width)
    DH_line.set_linewidth(dh_width)
    HE_line.set_linewidth(he_width)

    # Update scatter points with better depth ordering
    # Points should have slightly higher zorder than their connected lines
    point_zorder_boost = 5  # Points should be in front of their lines

    scatter_P._offsets3d = ([P[0]], [P[1]], [0])
    scatter_P.set_zorder(base_zorder - dist_p + point_zorder_boost)
    # Also adjust point size based on distance
    scatter_P.set_sizes([40 + max(0, 20 - dist_p*2)])

    scatter_Q._offsets3d = ([Q[0]], [Q[1]], [0])
    scatter_Q.set_zorder(base_zorder - dist_q + point_zorder_boost)
    scatter_Q.set_sizes([40 + max(0, 20 - dist_q*2)])

    scatter_D._offsets3d = ([D[0]], [D[1]], [0])
    scatter_D.set_zorder(base_zorder - dist_d + point_zorder_boost)
    scatter_D.set_sizes([40 + max(0, 20 - dist_d*2)])

    scatter_E._offsets3d = ([E[0]], [E[1]], [0])
    scatter_E.set_zorder(base_zorder - dist_e + point_zorder_boost)
    scatter_E.set_sizes([40 + max(0, 20 - dist_e*2)])

    scatter_H._offsets3d = ([H[0]], [H[1]], [H[2]])
    scatter_H.set_zorder(base_zorder - dist_h + point_zorder_boost)
    scatter_H.set_sizes([40 + max(0, 20 - dist_h*2)])

    # Update text positions with better depth ordering
    # Text should be in front of their corresponding points
    text_zorder_boost = 10  # Text should be in front of their points

    text_P.set_position((P[0], P[1]))
    text_P.set_3d_properties(0)
    text_P.set_zorder(base_zorder - dist_p + text_zorder_boost)
    # Adjust text color opacity based on distance (closer = more opaque)
    text_P.set_alpha(min(1.0, max(0.5, 1.2 - dist_p/5)))

    text_Q.set_position((Q[0], Q[1]))
    text_Q.set_3d_properties(0)
    text_Q.set_zorder(base_zorder - dist_q + text_zorder_boost)
    text_Q.set_alpha(min(1.0, max(0.5, 1.2 - dist_q/5)))

    # Return all dynamic elements including new trace lines
    return [op_line, oq_line, PD_line, QE_line, DH_line, HE_line,
            scatter_P, scatter_Q, scatter_D, scatter_E, scatter_H, text_P, text_Q] + trace_lines

# Set a camera angle that provides good visibility
ax.view_init(elev=25, azim=45)  # Better initial viewing angle

# Set axis limits to maintain consistent view
ax.set_xlim(-a*1.2, a*1.2)
ax.set_ylim(-a*1.2, a*1.2)
ax.set_zlim(0, h*1.2)

# Improve grid and background for better depth perception
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='--', alpha=0.3)

# Add a subtle ground plane for better depth perception
xx, yy = np.meshgrid(np.linspace(-a*1.2, a*1.2, 2), np.linspace(-a*1.2, a*1.2, 2))
zz = np.zeros_like(xx) - 0.01  # Slightly below z=0
ground = ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1, shade=True)
# Can't set zorder directly on plot_surface, but it should be drawn first anyway

# Create animation with proper interval for smooth rendering
# Using blit=True for better performance, but returning all elements to ensure they're properly updated
ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)

# Enable automatic depth sorting
ax.computed_zorder = True

# Save using FFMpegWriter if possible, else GIF
try:
    writer = FFMpegWriter(fps=30)
    ani.save('angle_POQ_gradient.mp4', writer=writer)
    print("Animation saved as MP4.")
except Exception as e:
    print(f"FFMpegWriter unavailable ({e}); saving as GIF.")
    ani.save('angle_POQ_gradient.gif', writer='pillow', fps=30)

plt.show()