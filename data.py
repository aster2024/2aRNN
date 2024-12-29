"""
Generate training data for context-dependent decision-making tasks.

Reference:
Mante, V., Sussillo, D., Shenoy, K. V. & Newsome, W. T. Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature 503, 78-84 (2013).

"""
import numpy as np

default_timing = [
    300, # fixation
    1000, # stimulus
    900,    # delay
    500,   # response
]

def gen_data(n_trials, dt=20, timing=default_timing, noise=0.05, seed=0):
    """
    Generate training data for context-dependent decision-making tasks.

    Args:
        n_trials (int): Number of trials.
        dt (float): Size of time step.
        timing (list): Timing of different task period.
        seed (int): Random seed.

    Returns:
        x (np.ndarray): Input data of shape (n_trials, time_steps, 5).
        y (np.ndarray): Output data of shape (n_trials, time_steps, 2).
        metadata (dict): Metadata including the context, stimulus, action, and timing.

    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Generate context data
    context = rng.randint(2, size=n_trials)

    # Generate stimulus data
    stim1_coh = rng.uniform(-1, 1, n_trials).astype(np.float32)
    stim2_coh = rng.uniform(-1, 1, n_trials).astype(np.float32)
    stim_coh = np.where(context, stim2_coh, stim1_coh)

    # Generate action data
    action = np.sign(stim_coh)

    # timing
    n_steps_per_period = (np.asarray(timing) / dt).astype(int)
    n_steps_cumsum = np.cumsum(n_steps_per_period)[:-1]
    n_timing = {
        'fixation': slice(n_steps_cumsum[-1]),
        'stimulus': slice(n_steps_cumsum[0], n_steps_cumsum[1]),
        'delay': slice(n_steps_cumsum[1], n_steps_cumsum[2]),
        'response': slice(n_steps_cumsum[-1], None),
        }
    # Generate input data
    n_steps = np.sum(n_steps_per_period)

    x = np.zeros((n_trials, n_steps, 5), dtype=np.float32)
    # add fixation
    x[:, n_timing['fixation'], 0] = 1
    # add stimulus
    x[:, n_timing['stimulus'], 1] = stim1_coh.reshape(-1, 1)
    x[:, n_timing['stimulus'], 2] = stim2_coh.reshape(-1, 1)
    # add noise
    x[:, n_timing['stimulus'], 1:3] += rng.normal(
        0, noise*np.sqrt(dt), size=(n_trials, n_steps_per_period[1], 2)).astype(np.float32)
    # add context cue
    x[context==0, :, 3] = 1
    x[context==1, :, 4] = 1

    # Generate output data
    y = np.zeros((n_trials, n_steps, 2), dtype=np.float32)
    # add fixation
    y[:, n_timing['fixation'], 0] = 1
    y[:, n_timing['response'], 1] = action.reshape(-1, 1)

    metadata = {
        'stim1_coh': stim1_coh,
        'stim2_coh': stim2_coh,
        'stim_coh':  stim_coh,
        'ctx':       context,
        'action':    action,
        'timing':    n_timing,
        }

    return x, y, metadata

def gen_data_fixed_stim(n_trials, fixed_stim, ctx, dt=20, timing=[300, 1000, 900, 500], noise=0.05, seed=0):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    context = np.full(n_trials, ctx)

    if ctx == 0:
        stim1_coh = np.full(n_trials, fixed_stim)
        stim2_coh = rng.uniform(-1, 1, n_trials).astype(np.float32)
    else:
        stim1_coh = rng.uniform(-1, 1, n_trials).astype(np.float32)
        stim2_coh = np.full(n_trials, fixed_stim)

    stim_coh = np.where(context, stim2_coh, stim1_coh)
    action = np.sign(stim_coh)

    n_steps_per_period = (np.asarray(timing) / dt).astype(int)
    n_steps_cumsum = np.cumsum(n_steps_per_period)[:-1]
    n_timing = {
        'fixation': slice(n_steps_cumsum[-1]),
        'stimulus': slice(n_steps_cumsum[0], n_steps_cumsum[1]),
        'delay': slice(n_steps_cumsum[1], n_steps_cumsum[2]),
        'response': slice(n_steps_cumsum[-1], None),
    }

    n_steps = np.sum(n_steps_per_period)

    x = np.zeros((n_trials, n_steps, 5), dtype=np.float32)
    x[:, n_timing['fixation'], 0] = 1
    x[:, n_timing['stimulus'], 1] = stim1_coh.reshape(-1, 1)
    x[:, n_timing['stimulus'], 2] = stim2_coh.reshape(-1, 1)
    x[:, n_timing['stimulus'], 1:3] += rng.normal(0, noise * np.sqrt(dt),
                                                  size=(n_trials, n_steps_per_period[1], 2)).astype(np.float32)
    x[context == 0, :, 3] = 1
    x[context == 1, :, 4] = 1

    y = np.zeros((n_trials, n_steps, 2), dtype=np.float32)
    y[:, n_timing['fixation'], 0] = 1
    y[:, n_timing['response'], 1] = action.reshape(-1, 1)

    metadata = {
        'stim1_coh': stim1_coh,
        'stim2_coh': stim2_coh,
        'stim_coh': stim_coh,
        'ctx': context,
        'action': action,
        'timing': n_timing,
    }

    return x, y, metadata

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x, y, metadata = gen_data(100)
    print(x.shape, y.shape)
    print(metadata.keys())
    fig, ax = plt.subplots(1, 2, figsize=(12, 2.5))
    cb1 = ax[0].pcolormesh(x[2].T)
    cb2 = ax[1].pcolormesh(y[2].T)
    plt.colorbar(cb1, ax=ax[0])
    plt.colorbar(cb2, ax=ax[1])
    ax[0].set_title('Input data')
    ax[1].set_title('Output data')
    ax[0].set_xlabel('Time steps')
    ax[1].set_xlabel('Time steps')
    ax[0].set_ylabel('Input features')
    ax[1].set_ylabel('Output features')
    fig.savefig('data.png', bbox_inches='tight')