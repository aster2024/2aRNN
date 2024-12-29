import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import SingleAreaRNN, TwoAreaRNN
from data import gen_data
import os
from datetime import datetime


def train_model(model, device, seed):
    n_epochs = 100
    n_trials_per_epoch = 100
    batch_size = 20
    num_batches = n_trials_per_epoch // batch_size
    task_timing = [300, 1000, 900, 500]

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_list = []

    for epoch in range(n_epochs):
        task_timing_ = task_timing.copy()
        task_timing_[2] = np.random.randint(300, 1500)

        x, y, metadata = gen_data(n_trials_per_epoch, timing=task_timing_)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        loss_buff = 0
        for batch in range(num_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            output = model(x[batch_slice])
            loss = criterion(output, y[batch_slice])
            loss_buff += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss_buff / num_batches)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss {loss_list[-1]:.6f}')

    return loss_list


def evaluate_model(model, device):
    task_timing = [300, 1000, 900, 500]
    model.eval()
    with torch.no_grad():
        x, y, metadata = gen_data(100, timing=task_timing)
        x = torch.from_numpy(x).to(device)
        outputs, hs = model(x, return_hidden=True)
        outputs = outputs.cpu().numpy()

        if isinstance(hs, tuple):  # TwoAreaRNN
            hs = tuple(h.cpu().numpy() for h in hs)
        else:  # SingleAreaRNN
            hs = hs.cpu().numpy()

        decisions = np.sign(outputs[:, -1, 1])
        accuracy = np.mean(decisions == metadata['action'])

    return accuracy, hs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs('fig', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    models_config = {
        '1aRNN': (SingleAreaRNN, {'input_size': 5, 'hidden_size': 100, 'output_size': 2}),
        '2aRNN': (TwoAreaRNN, {'input_size': 5, 'hidden_size': 100, 'output_size': 2})
    }

    all_accuracies = {model_name: [] for model_name in models_config.keys()}

    for seed in range(5):
        print(f"\nTraining models with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        plt.figure(figsize=(10, 6))

        for model_name, (model_class, model_args) in models_config.items():
            print(f"\nTraining {model_name}")

            model = model_class(**model_args).to(device)
            if model_name == '2aRNN':
                model.alpha2 = model.alpha1 / 10

            loss_list = train_model(model, device, seed)
            accuracy, hs = evaluate_model(model, device)

            all_accuracies[model_name].append(accuracy)

            plt.semilogy(loss_list, label=model_name)
            save_path = os.path.join('model_bonus', f'{model_name}_seed{seed}_acc{accuracy:.3f}.pt')
            torch.save(model.state_dict(), save_path)

        plt.title(f'Loss Curves (Seed {seed})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'fig_bonus/loss_curves_seed{seed}.png')
        plt.close()

    # Boxplot
    # plt.figure(figsize=(8, 6))
    # plt.boxplot([all_accuracies['1aRNN'], all_accuracies['2aRNN']],
    #             labels=['1aRNN', '2aRNN'])
    # plt.title('Model Accuracies')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.savefig('fig_bonus/accuracy_boxplot.png')
    # plt.close()

    with open('fig_bonus/accuracies.txt', 'w') as f:
        for model_name, accs in all_accuracies.items():
            f.write(f'{model_name}:\n')
            for seed, acc in enumerate(accs):
                f.write(f'  Seed {seed}: {acc:.4f}\n')
            f.write(f'  Mean: {np.mean(accs):.4f}\n')
            f.write(f'  Std: {np.std(accs):.4f}\n\n')


if __name__ == '__main__':
    main()
