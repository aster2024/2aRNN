import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('Loss Curves Comparison Across Seeds', fontsize=16)

for seed in range(5):
    img = mpimg.imread(f'fig/loss_curves_seed{seed}.png')
    axes[seed].imshow(img)
    axes[seed].set_title(f'Seed {seed}')
    axes[seed].axis('off')

plt.tight_layout()
plt.savefig('fig/all_seeds_loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
