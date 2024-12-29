import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from model import TwoAreaRNN
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from data import gen_data, gen_data_fixed_stim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_rnn_hidden_states(model, x):
    with torch.no_grad():
        _, (h1s, h2s) = model(x, return_hidden=True)
    return h1s, h2s


def linear_regression(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def plot_trajectory_in_space(h_list, beta_x, beta_y, stim_values, x_label, y_label, title, ctx, area, fixed_points=None,
                             save_path=None):
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

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{title} - Area {area}", fontsize=12)

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
    plt.show()


def find_fixed_points(model, ctx, timing, dt=20, n_points=50, n_iters=1000, learning_rate=0.1, batch_size=50, seed=0,
                      tolerance=1e-3):
    model_noise = model.noise
    model.noise = 0.0
    model.requires_grad_(False)

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    n_steps_per_period = (np.asarray(timing) / dt).astype(int)
    n_steps_cumsum = np.cumsum(n_steps_per_period)[:-1]
    n_timing = {
        'fixation': slice(n_steps_cumsum[-1]),
        'stimulus': slice(n_steps_cumsum[0], n_steps_cumsum[1]),
        'delay': slice(n_steps_cumsum[1], n_steps_cumsum[2]),
        'response': slice(n_steps_cumsum[-1], None),
    }
    n_steps = np.sum(n_steps_per_period)
    fixed_points_1 = []
    fixed_points_2 = []

    for batch_start in tqdm(range(0, n_points, batch_size), desc=f'Context {ctx} fixed points'):
        batch_end = min(batch_start + batch_size, n_points)
        batch_size_current = batch_end - batch_start

        h1 = torch.randn(batch_size_current, model.hidden_size, device=device, requires_grad=True)
        h2 = torch.randn(batch_size_current, model.hidden_size, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([h1, h2], lr=learning_rate)

        for _ in range(n_iters):
            optimizer.zero_grad()

            x = torch.zeros(batch_size_current, n_steps, 5, device=device)
            x[:, n_timing['fixation'], 0] = 1  # fixation
            x[:, :, 3 if ctx == 0 else 4] = 1  # context

            h1_next, h2_next = model.get_final_state(x, h1, h2)
            loss = torch.sum((h1_next - h1) ** 2, dim=1) + torch.sum((h2_next - h2) ** 2, dim=1)

            if _ % 100 == 0:
                print(f'Loss: {loss.mean().item()}')

            total_loss = loss.sum()
            total_loss.backward()
            optimizer.step()

        converged = loss < tolerance
        fixed_points_1.extend(h1[converged].detach().cpu().numpy())
        fixed_points_2.extend(h2[converged].detach().cpu().numpy())

    model.noise = model_noise
    print(f'Found {len(fixed_points_1)} fixed points for Area 1')
    print(f'Found {len(fixed_points_2)} fixed points for Area 2')
    return np.array(fixed_points_1), np.array(fixed_points_2)


def main():
    model = TwoAreaRNN(input_size=5, hidden_size=100, output_size=2).to(device)
    model.load_state_dict(torch.load('model/2aRNN_seed3_acc1.000.pt'))
    model.eval()

    stim_values = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]
    n_trials_per_stim = 500
    task_timing = [300, 1000, 900, 500]

    for ctx in [0, 1]:
        fixed_points_1, fixed_points_2 = find_fixed_points(model, ctx, task_timing)
        h1_list = []
        h2_list = []
        h1_all = []
        h2_all = []
        stim1_coh_all = []
        stim2_coh_all = []
        choice_all = []

        x, _, metadata = gen_data(2000, timing=task_timing)
        x = torch.from_numpy(x).to(device)
        h1s, h2s = get_rnn_hidden_states(model, x)

        for t in range(h1s.shape[1]):
            h1_all.append(h1s[:, t, :].cpu().numpy())
            h2_all.append(h2s[:, t, :].cpu().numpy())
            stim1_coh_all.append(metadata['stim1_coh'])
            stim2_coh_all.append(metadata['stim2_coh'])
            choice_all.append(metadata['action'])

        # Concatenate all data
        h1_all = np.concatenate(h1_all)
        h2_all = np.concatenate(h2_all)
        stim1_coh_all = np.concatenate(stim1_coh_all)
        stim2_coh_all = np.concatenate(stim2_coh_all)
        choice_all = np.concatenate(choice_all)

        for stim in stim_values:
            x, _, metadata = gen_data_fixed_stim(n_trials_per_stim, stim, ctx, timing=task_timing)
            x = torch.from_numpy(x).to(device)
            h1s, h2s = get_rnn_hidden_states(model, x)
            h1s_sampled = h1s[:, ::5, :]  # Sample every 5 time steps
            h2s_sampled = h2s[:, ::5, :]  # Sample every 5 time steps
            h1_avg = h1s_sampled.mean(axis=0)  # Average over trials
            h2_avg = h2s_sampled.mean(axis=0)  # Average over trials
            h1_list.append(h1_avg.cpu().numpy())
            h2_list.append(h2_avg.cpu().numpy())

        for area, h_list, h_all, fixed_points in [(1, h1_list, h1_all, fixed_points_1),
                                                  (2, h2_list, h2_all, fixed_points_2)]:
            beta_1 = linear_regression(h_all, stim1_coh_all)
            beta_2 = linear_regression(h_all, stim2_coh_all)
            beta_3 = linear_regression(h_all, choice_all)

            # Orthogonalization
            Q, R = qr(np.stack([beta_1, beta_2, beta_3], axis=-1))
            beta_1_prime, beta_2_prime, beta_3_prime = Q[:, 0], Q[:, 1], Q[:, 2]

            plot_trajectory_in_space(h_list, beta_3_prime, beta_1_prime, stim_values, 'Choice', 'Motion',
                                     f'{"Motion" if ctx == 0 else "Colour"} Context', ctx, area,
                                     fixed_points=fixed_points,
                                     save_path=f'fig/context_{ctx}_choice_motion_area{area}.png')

            plot_trajectory_in_space(h_list, beta_3_prime, beta_2_prime, stim_values, 'Choice', 'Colour',
                                     f'{"Motion" if ctx == 0 else "Colour"} Context', ctx, area,
                                     fixed_points=fixed_points,
                                     save_path=f'fig/context_{ctx}_choice_colour_area{area}.png')

        # Sample and plot trajectories from the other context
        other_ctx = 1 - ctx
        h1_list_other = []
        h2_list_other = []
        for stim in stim_values:
            x, _, metadata = gen_data_fixed_stim(n_trials_per_stim, stim, other_ctx, timing=task_timing)
            x = torch.from_numpy(x).to(device)
            h1s, h2s = get_rnn_hidden_states(model, x)
            h1s_sampled = h1s[:, ::5, :]  # Sample every 5 time steps
            h2s_sampled = h2s[:, ::5, :]  # Sample every 5 time steps
            h1_avg = h1s_sampled.mean(axis=0)  # Average over trials
            h2_avg = h2s_sampled.mean(axis=0)  # Average over trials
            h1_list_other.append(h1_avg.cpu().numpy())
            h2_list_other.append(h2_avg.cpu().numpy())

        for area, h_list_other, fixed_points in [(1, h1_list_other, fixed_points_1),
                                                 (2, h2_list_other, fixed_points_2)]:
            if ctx == 0:
                plot_trajectory_in_space(h_list_other, beta_3_prime, beta_2_prime, stim_values, 'Choice', 'Colour',
                                         f'Motion Context', ctx, area, fixed_points=fixed_points,
                                         save_path=f'fig/context_{ctx}_irrelevant_choice_colour_area{area}.png')
            else:
                plot_trajectory_in_space(h_list_other, beta_3_prime, beta_1_prime, stim_values, 'Choice', 'Motion',
                                         f'Colour Context', ctx, area, fixed_points=fixed_points,
                                         save_path=f'fig/context_{ctx}_irrelevant_choice_motion_area{area}.png')


if __name__ == '__main__':
    main()
