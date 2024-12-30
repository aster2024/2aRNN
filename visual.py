import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_in_space(h_list, beta_x, beta_y, stim_values, x_label, y_label, title, ctx, fixed_points=None,
                             save_path=None, xlim=None, ylim=None):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_facecolor('white')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    colors = plt.cm.Blues(np.linspace(0.3, 1, len(stim_values)))

    for h, stim, color in zip(h_list, stim_values, colors):
        h_proj = np.column_stack([h @ beta_x, h @ beta_y])
        ax.plot(h_proj[:, 0], h_proj[:, 1], color=color, linewidth=1.5)
        ax.scatter(h_proj[0, 0], h_proj[0, 1], color=color, marker='o', s=30, edgecolor='black', linewidth=0.5)
        ax.scatter(h_proj[-1, 0], h_proj[-1, 1], color=color, marker='s', s=30, edgecolor='black', linewidth=0.5)

    if fixed_points is not None and len(fixed_points) > 0:
        fixed_points_proj = np.column_stack([fixed_points @ beta_x, fixed_points @ beta_y])
        ax.plot(fixed_points_proj[:, 0], fixed_points_proj[:, 1], color='red', linewidth=2, linestyle='-',
                label='Line Attractor')
        ax.scatter(fixed_points_proj[:, 0], fixed_points_proj[:, 1], color='red', marker='x', s=50, linewidth=2)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)

    ax.text(min(ax.get_xlim()), min(ax.get_ylim()), 'Choice 1', ha='left', va='bottom', fontsize=10)
    ax.text(max(ax.get_xlim()), min(ax.get_ylim()), 'Choice 2', ha='right', va='bottom', fontsize=10)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='End'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Line Attractor'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=8, label='Fixed Points')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    context_text = 'Motion context' if ctx == 0 else 'Colour context'
    ax.text(0.98, 0.98, context_text, transform=ax.transAxes, ha='right', va='top', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()


def get_common_limits(h_list, beta_list):
    all_x = []
    all_y = []
    for h in h_list:
        for beta in beta_list:
            proj = h @ beta
            all_x.extend(proj)
            all_y.extend(proj)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min

    return [x_min - margin * x_range, x_max + margin * x_range], [y_min - margin * y_range, y_max + margin * y_range]


