import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from model import SingleAreaRNN
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from data import gen_data, gen_data_fixed_stim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_rnn_hidden_states(model, x):
    with torch.no_grad():
        _, hs = model(x, return_hidden=True)
    return hs


def linear_regression(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def plot_trajectory_in_space(h_list, beta_x, beta_y, stim_values, x_label, y_label, title, ctx, fixed_points=None, save_path=None):
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
        ax.plot(fixed_points_proj[:, 0], fixed_points_proj[:, 1], color='red', linewidth=2, linestyle='-', label='Line Attractor')
        ax.scatter(fixed_points_proj[:, 0], fixed_points_proj[:, 1], color='red', marker='x', s=50, linewidth=2)

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
    plt.show()


def orthogonalize_last_vector(vectors):
    v1 = vectors[:, 0] / np.linalg.norm(vectors[:, 0])
    v2 = vectors[:, 1] / np.linalg.norm(vectors[:, 1])

    proj_v3_on_v1 = np.dot(vectors[:, 2], v1) * v1
    proj_v3_on_v2 = np.dot(vectors[:, 2], v2) * v2

    v3_orthogonalized = vectors[:, 2] - proj_v3_on_v1 - proj_v3_on_v2

    v3_orthogonalized /= np.linalg.norm(v3_orthogonalized)

    return np.column_stack([v1, v2, v3_orthogonalized])


def logistic_regression_no_bias_with_gridsearch(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {'C': [0.001]}

    base_model = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=10000)

    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_

    print(f"Best C parameter: {best_model.C}")

    return best_model.coef_[0]


def find_fixed_points(model, ctx, timing, dt=20, n_points=20, n_iters=1000, learning_rate=0.1, batch_size=20, seed=0,
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
    fixed_points = []

    for batch_start in tqdm(range(0, n_points, batch_size), desc=f'Context {ctx} fixed points'):
        batch_end = min(batch_start + batch_size, n_points)
        batch_size_current = batch_end - batch_start

        h = torch.from_numpy(rng.uniform(-10, 10, (batch_size_current, model.hidden_size))).float().to(device).requires_grad_(True)

        optimizer = torch.optim.Adam([h], lr=learning_rate)

        for _ in range(n_iters):
            optimizer.zero_grad()

            x = torch.zeros(batch_size_current, n_steps, 5, device=device)
            x[:, n_timing['fixation'], 0] = 1  # fixation
            x[:, :, 3 if ctx == 0 else 4] = 1  # context

            h_next = model.get_final_state(x, h)
            loss = torch.sum((h_next - h) ** 2, dim=1)

            if _ % 100 == 0:
                print(f'Loss: {loss.mean().item()}')

            total_loss = loss.sum()
            total_loss.backward()
            optimizer.step()

        converged = loss < tolerance
        fixed_points.extend(h[converged].detach().cpu().numpy())

    model.noise = model_noise
    print(f'Found {len(fixed_points)} fixed points')
    return np.array(fixed_points)

def main():
    model = SingleAreaRNN(input_size=5, hidden_size=100, output_size=2).to(device)
    model.load_state_dict(torch.load('model/1aRNN_seed1_acc1.000.pt'))
    model.eval()

    stim_values = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]
    n_trials_per_stim = 500
    task_timing = [300, 1000, 900, 500]

    for ctx in [0, 1]:
        fixed_points = find_fixed_points(model, ctx, task_timing)
        h_list = []
        h_all = []
        stim1_coh_all = []
        stim2_coh_all = []
        choice_all = []

        x, _, metadata = gen_data(2000, timing=task_timing)
        x = torch.from_numpy(x).to(device)
        hs = get_rnn_hidden_states(model, x)

        for t in range(hs.shape[1]):
            h_all.append(hs[:, t, :].cpu().numpy())
            stim1_coh_all.append(metadata['stim1_coh'])
            stim2_coh_all.append(metadata['stim2_coh'])
            choice_all.append(metadata['action'])

        # Concatenate all data
        h_all = np.concatenate(h_all)
        stim1_coh_all = np.concatenate(stim1_coh_all)
        stim2_coh_all = np.concatenate(stim2_coh_all)
        choice_all = np.concatenate(choice_all)

        for stim in stim_values:
            x, _, metadata = gen_data_fixed_stim(n_trials_per_stim, stim, ctx, timing=task_timing)
            x = torch.from_numpy(x).to(device)
            hs = get_rnn_hidden_states(model, x)
            hs_sampled = hs[:, ::5, :]  # Sample every 5 time steps
            h_avg = hs_sampled.mean(axis=0)  # Average over trials
            h_list.append(h_avg.cpu().numpy())

            # Collect all data points
            # h_all.append(hs_sampled[:, -1, :].cpu().numpy())  # Last time step
            # stimulus_start = metadata['timing']['stimulus'].start
            # stimulus_stop = metadata['timing']['stimulus'].stop
            # h_avg = hs[:, stimulus_start:stimulus_stop, :].mean(dim=1).cpu().numpy()
            # h_all.append(h_avg)
            # n_samples = 20
            # total_timesteps = hs.shape[1]
            # sampled_timesteps = np.sort(np.random.choice(total_timesteps, n_samples, replace=False))


        # Perform logistic regression without bias on all data points, with grid search
        # print("Fitting beta_1...")
        # beta_1 = logistic_regression_no_bias_with_gridsearch(h_all, (stim1_coh_all > 0).astype(int))
        # print("Fitting beta_2...")
        # beta_2 = logistic_regression_no_bias_with_gridsearch(h_all, (stim2_coh_all > 0).astype(int))
        # print("Fitting beta_3...")
        # beta_3 = logistic_regression_no_bias_with_gridsearch(h_all, (choice_all > 0).astype(int))

        beta_1 = linear_regression(h_all, stim1_coh_all)
        beta_2 = linear_regression(h_all, stim2_coh_all)
        beta_3 = linear_regression(h_all, choice_all)

        # beta_1_prime = beta_1 / np.linalg.norm(beta_1)
        # beta_2_prime = beta_2 / np.linalg.norm(beta_2)
        # beta_3_prime = beta_3 / np.linalg.norm(beta_3)
        # beta_1_prime = beta_1
        # beta_2_prime = beta_2
        # beta_3_prime = beta_3

        # Orthogonalization
        Q, R = qr(np.stack([beta_1, beta_2, beta_3], axis=-1))
        beta_1_prime, beta_2_prime, beta_3_prime = Q[:, 0], Q[:, 1], Q[:, 2]
        # Orthogonalize the last vector
        # vectors = np.stack([beta_1, beta_2, beta_3], axis=-1)
        # orthogonalized_vectors = orthogonalize_last_vector(vectors)
        # beta_1_prime, beta_2_prime, beta_3_prime = orthogonalized_vectors[:, 0], orthogonalized_vectors[:,
        #                                                                          1], orthogonalized_vectors[:, 2]

        plot_trajectory_in_space(h_list, beta_3_prime, beta_1_prime, stim_values, 'Choice', 'Motion', f'{"Motion" if ctx == 0 else "Colour"} Context',
                                 ctx, fixed_points=fixed_points, save_path=f'fig/context_{ctx}_choice_motion.png')
        plot_trajectory_in_space(h_list, beta_3_prime, beta_2_prime, stim_values, 'Choice', 'Colour', f'{"Motion" if ctx == 0 else "Colour"} Context',
                                 ctx, fixed_points=fixed_points, save_path=f'fig/context_{ctx}_choice_colour.png')

        # Sample and plot trajectories from the other context
        other_ctx = 1 - ctx
        h_list_other = []
        for stim in stim_values:
            x, _, metadata = gen_data_fixed_stim(n_trials_per_stim, stim, other_ctx, timing=task_timing)
            x = torch.from_numpy(x).to(device)
            hs = get_rnn_hidden_states(model, x)
            hs_sampled = hs[:, ::5, :]  # Sample every 5 time steps
            h_avg = hs_sampled.mean(axis=0)  # Average over trials
            h_list_other.append(h_avg.cpu().numpy())

        if ctx == 0:
            plot_trajectory_in_space(h_list_other, beta_3_prime, beta_2_prime, stim_values, 'Choice', 'Colour',f'Motion Context',
                                     ctx, fixed_points=fixed_points, save_path=f'fig/context_{ctx}_irrelevant_choice_colour.png')
        else:
            plot_trajectory_in_space(h_list_other, beta_3_prime, beta_1_prime, stim_values, 'Choice', 'Motion',f'Colour Context',
                                     ctx, fixed_points=fixed_points, save_path=f'fig/context_{ctx}_irrelevant_choice_motion.png')

if __name__ == '__main__':
    main()
