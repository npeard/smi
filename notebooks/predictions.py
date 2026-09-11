import marimo

__generated_with = '0.20.4'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    return LineCollection, LogNorm, mo, np, plt, torch


@app.cell
def _():
    from smi.analysis.datasets import get_data_loaders
    from smi.analysis.generate_data import generate_synthetic_test_data
    from smi.analysis.lit_module import LitModule
    from smi.analysis.synthetic_lit_module import DEFAULT_WAVELENGTHS_NM
    from smi.redpitaya.redpitaya_config import RedPitayaConfig

    ACQ_SAMPLE_RATE = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
    CHANNEL_NAMES = ['PD1 (635 nm)', 'PD2 (675 nm)', 'PD3 (515 nm)']
    return (
        ACQ_SAMPLE_RATE,
        CHANNEL_NAMES,
        DEFAULT_WAVELENGTHS_NM,
        LitModule,
        generate_synthetic_test_data,
        get_data_loaders,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Model Predictions Viewer

    Load a trained checkpoint and visualize predictions against target data.
    Supports both real HDF5 datasets and synthetic test data.
    """)


@app.cell
def _():
    freespace_datapath = '/Users/nolanpeard/Documents/Projects/smi/smi/analysis/data/free-space-synchro_10k.h5'
    mmfiber_datapath = '/Users/nolanpeard/Documents/Projects/smi/smi/analysis/data/mmfiber-synchro_10k.h5'

    best_freespace_model = '/Users/nolanpeard/Documents/Projects/smi/smi/analysis/models/checkpoints/4puqewmv_epoch=199-val_total_unweighted_loss=0.0000.ckpt'
    best_fiber_model = '/Users/nolanpeard/Documents/Projects/smi/smi/analysis/models/checkpoints/bfzctg24_epoch=199-val_total_unweighted_loss=0.0000.ckpt'
    return best_fiber_model, mmfiber_datapath


@app.cell
def _(best_fiber_model, mmfiber_datapath, mo):
    checkpoint_path = mo.ui.text(label='Checkpoint path', value=best_fiber_model)
    dataset_path = mo.ui.text(
        label="Dataset path (or 'synthetic')", value=mmfiber_datapath
    )
    batch_size = mo.ui.number(label='Batch size', value=128, start=1, stop=512, step=1)
    sample_index = mo.ui.slider(label='Sample index', start=0, stop=9, value=3)
    mo.vstack([checkpoint_path, dataset_path, batch_size, sample_index])
    return batch_size, checkpoint_path, dataset_path, sample_index


@app.cell
def _(LitModule, checkpoint_path, mo, torch):
    mo.stop(not checkpoint_path.value, mo.md('Enter a checkpoint path above.'))
    _map_location = 'cpu' if not torch.cuda.is_available() else None
    # Manual loading to strip torch.compile _orig_mod. key prefixes
    _ckpt = torch.load(
        checkpoint_path.value, weights_only=False, map_location=_map_location
    )
    ckpt_hparams = _ckpt.get('hyper_parameters', {})
    _state_dict = {
        k.replace('._orig_mod.', '.'): v for k, v in _ckpt['state_dict'].items()
    }
    model = LitModule(**ckpt_hparams)
    model.load_state_dict(_state_dict)
    in_channels = model.hparams['model_hparams']['in_channels']
    return ckpt_hparams, in_channels, model


@app.cell
def _(
    DEFAULT_WAVELENGTHS_NM,
    batch_size,
    ckpt_hparams,
    dataset_path,
    generate_synthetic_test_data,
    get_data_loaders,
    in_channels,
):
    if dataset_path.value == 'synthetic':
        _wavelengths_nm = ckpt_hparams.get(
            'wavelengths_nm', DEFAULT_WAVELENGTHS_NM[:in_channels]
        )
        _start_freq = ckpt_hparams.get('start_freq', 1.0)
        _end_freq = ckpt_hparams.get('end_freq', 1000.0)
        _max_displacement_um = ckpt_hparams.get('max_displacement_um', 5.0)
        test_loader = generate_synthetic_test_data(
            num_samples=batch_size.value * 4,
            wavelengths_nm=_wavelengths_nm,
            batch_size=batch_size.value,
            start_freq=_start_freq,
            end_freq=_end_freq,
            max_displacement_um=_max_displacement_um,
        )
    else:
        _, _, test_loader = get_data_loaders(
            dataset_path.value, batch_size=batch_size.value
        )
    return (test_loader,)


@app.cell
def _(model, test_loader, torch):
    # Run predictions on a single batch only
    model.eval()
    _batch = next(iter(test_loader))
    with torch.no_grad():
        _result = model.predict_step(_batch, 0)

    velocity_hat = _result[0].cpu().numpy()
    velocity_target = _result[1].cpu().numpy()
    displacement_hat = _result[2].cpu().numpy()
    displacement_target = _result[3].cpu().numpy()
    signals = _result[4].cpu().numpy()

    # Batch-level MSE
    _mse = torch.nn.MSELoss()
    batch_velocity_mse = _mse(
        torch.tensor(velocity_hat), torch.tensor(velocity_target)
    ).item()
    batch_displacement_mse = _mse(
        torch.tensor(displacement_hat), torch.tensor(displacement_target)
    ).item()
    return (
        batch_displacement_mse,
        batch_velocity_mse,
        displacement_hat,
        displacement_target,
        signals,
        velocity_hat,
        velocity_target,
    )


@app.cell
def _(np, torch):
    def compute_input_gradients(mdl, sigs):
        """Compute gradients of model output with respect to input signals.

        For each sample, computes d(output[t])/d(input) for all timesteps t.
        Returns the sum of absolute gradients across all output timesteps,
        showing which input points have the strongest overall influence.

        Args:
            mdl: The trained model
            sigs: Input signals tensor [batch_size, num_channels, signal_length]

        Returns:
            Gradients array [batch_size, num_channels, signal_length]
        """
        mdl.training = True
        signals_input = torch.tensor(
            sigs.detach().cpu().numpy(),
            dtype=torch.float32,
            requires_grad=True,
            device='cpu',
        )
        with torch.enable_grad():
            output = mdl(signals_input)
            grad_outputs = torch.ones_like(output)
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=signals_input,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]
        return np.abs(gradients.detach().numpy())

    return (compute_input_gradients,)


@app.cell
def _(compute_input_gradients, model, signals, torch):
    _model_cpu = model.cpu()
    _signals_tensor = torch.tensor(signals, dtype=torch.float32, requires_grad=True)
    abs_gradients = compute_input_gradients(_model_cpu, _signals_tensor)

    _vmean = abs_gradients.mean()
    _vstd = abs_gradients.std()
    grad_vmin = 0
    grad_vmax = _vmean + 2 * _vstd
    return abs_gradients, grad_vmax, grad_vmin


@app.cell
def _(mo):
    mo.md(r"""
    ## Raw Data
    """)


@app.cell
def _(ACQ_SAMPLE_RATE, CHANNEL_NAMES, np, plt, sample_index, signals, velocity_target):
    _i = sample_index.value
    _batch_size, _num_channels, _signal_length = signals.shape
    _time_ms = np.arange(_signal_length) / ACQ_SAMPLE_RATE * 1000
    _chan_names = [
        CHANNEL_NAMES[_j] if _j < len(CHANNEL_NAMES) else f'CH{_j + 1}'
        for _j in range(_num_channels)
    ]

    _fig, _axs = plt.subplots(
        _num_channels, 1, figsize=(18, 3 * _num_channels), sharex=True
    )
    if _num_channels == 1:
        _axs = [_axs]
    for _j in range(_num_channels):
        _axs[_j].plot(_time_ms, signals[_i, _j, :], label='Signal')
        _axs[_j].set_title(f'Input Signal - {_chan_names[_j]}')
        _axs[_j].set_ylabel('Amplitude')
        _axs[_j].grid(True, alpha=0.3)
        # Overlay velocity on split y-axis
        _ax2 = _axs[_j].twinx()
        _ax2.plot(
            _time_ms,
            velocity_target[_i],
            color='#972a19',
            alpha=1.0,
            label='Velocity',
            lw=3,
        )
        _ax2.set_ylabel('Velocity (umm/s)', color='#972a19')
        _ax2.tick_params(axis='y', labelcolor='#972a19')
    _axs[-1].set_xlabel('Time (ms)')
    # _fig.suptitle(f'Raw Input Signals - Sample {_i + 1}')
    plt.tight_layout()
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## Predictions & Residuals
    """)


@app.cell
def _(
    ACQ_SAMPLE_RATE,
    displacement_hat,
    displacement_target,
    np,
    plt,
    sample_index,
    signals,
    torch,
    velocity_hat,
    velocity_target,
):
    _i = sample_index.value
    _signal_length = signals.shape[2]
    _time_ms = np.arange(_signal_length) / ACQ_SAMPLE_RATE * 1000
    _mse = torch.nn.MSELoss()
    _v_mse = _mse(
        torch.tensor(velocity_hat[_i]), torch.tensor(velocity_target[_i])
    ).item()
    _d_mse = _mse(
        torch.tensor(displacement_hat[_i]), torch.tensor(displacement_target[_i])
    ).item()
    _v_rmse = np.sqrt(_v_mse)
    _d_rmse = np.sqrt(_d_mse)
    _v_residual = velocity_hat[_i] - velocity_target[_i]
    _d_residual = displacement_hat[_i] - displacement_target[_i]

    _fig, _axs = plt.subplots(4, 1, figsize=(18, 10), sharex=True)

    # Velocity predictions
    _axs[0].plot(_time_ms, velocity_target[_i], label='Target', color='#1b3c9e')
    _axs[0].plot(
        _time_ms, velocity_hat[_i], label='Predicted', color='#d95f02', linestyle='--'
    )
    _axs[0].set_title(f'Velocity (RMSE: {_v_rmse:.1f} umm/s)')
    _axs[0].set_ylabel('Velocity (umm/s)')
    _axs[0].grid(True, alpha=0.3)
    _axs[0].legend(loc='upper right')

    # Velocity residual
    _axs[1].plot(_time_ms, _v_residual, label='Residual', color='#7570b3')
    _axs[1].axhline(y=0, color='black', linestyle=':')
    _axs[1].set_title(f'Velocity Residual (MSE: {_v_mse:.2e})')
    _axs[1].set_ylabel('Residual (umm/s)')
    _axs[1].grid(True, alpha=0.3)
    _axs[1].legend(loc='upper right')

    # Displacement predictions
    _axs[2].plot(_time_ms, displacement_target[_i], label='Target', color='#1b3c9e')
    _axs[2].plot(
        _time_ms,
        displacement_hat[_i],
        label='Predicted',
        color='#d95f02',
        linestyle='--',
    )
    _axs[2].set_title(f'Displacement (RMSE: {_d_rmse:.4f} umm)')
    _axs[2].set_ylabel('Displacement (umm)')
    _axs[2].grid(True, alpha=0.3)
    _axs[2].legend(loc='upper right')

    # Displacement residual
    _axs[3].plot(_time_ms, _d_residual, label='Residual', color='#7570b3')
    _axs[3].axhline(y=0, color='black', linestyle=':')
    _axs[3].set_title(f'Displacement Residual (MSE: {_d_mse:.2e})')
    _axs[3].set_ylabel('Residual (umm)')
    _axs[3].grid(True, alpha=0.3)
    _axs[3].legend(loc='upper right')

    _axs[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ## Combined View
    """)


@app.cell
def _(
    ACQ_SAMPLE_RATE,
    CHANNEL_NAMES,
    LineCollection,
    abs_gradients,
    batch_displacement_mse,
    batch_velocity_mse,
    displacement_hat,
    displacement_target,
    grad_vmax,
    grad_vmin,
    np,
    plt,
    sample_index,
    signals,
    torch,
    velocity_hat,
    velocity_target,
):
    _i = sample_index.value
    _batch_size, _num_channels, _signal_length = signals.shape
    _time_ms = np.arange(_signal_length) / ACQ_SAMPLE_RATE * 1000
    _mse = torch.nn.MSELoss()
    _v_mse = _mse(
        torch.tensor(velocity_hat[_i]), torch.tensor(velocity_target[_i])
    ).item()
    _d_mse = _mse(
        torch.tensor(displacement_hat[_i]), torch.tensor(displacement_target[_i])
    ).item()
    _v_residual = velocity_hat[_i] - velocity_target[_i]
    _d_residual = displacement_hat[_i] - displacement_target[_i]

    _chan_names = [
        CHANNEL_NAMES[_j] if _j < len(CHANNEL_NAMES) else f'CH{_j + 1}'
        for _j in range(_num_channels)
    ]

    _fig, _axs = plt.subplots(4 + _num_channels, 1, figsize=(12, 12), sharex=True)

    # Velocity predictions
    _axs[0].plot(_time_ms, velocity_target[_i], label='Target Velocity', color='blue')
    _axs[0].plot(
        _time_ms,
        velocity_hat[_i],
        label='Predicted Velocity',
        color='red',
        linestyle='--',
    )
    _axs[0].set_title(f'Velocity (MSE: {_v_mse:.2e})')
    _axs[0].set_ylabel('Velocity (umm/s)')
    _axs[0].grid(True, alpha=0.3)
    _axs[0].legend(loc='upper right')

    # Velocity residuals
    _axs[1].plot(_time_ms, _v_residual, label='Velocity Residual', color='purple')
    _axs[1].axhline(y=0, color='black', linestyle=':', alpha=0.5)
    _axs[1].set_title(f'Velocity Residual (MSE: {_v_mse:.2e})')
    _axs[1].set_ylabel('Residual (umm/s)')
    _axs[1].grid(True, alpha=0.3)
    _axs[1].legend(loc='upper right')

    # Displacement predictions
    _axs[2].plot(
        _time_ms, displacement_target[_i], label='Target Displacement', color='green'
    )
    _axs[2].plot(
        _time_ms,
        displacement_hat[_i],
        label='Predicted Displacement',
        color='orange',
        linestyle='--',
    )
    _axs[2].set_title(f'Displacement (MSE: {_d_mse:.2e})')
    _axs[2].set_ylabel('Displacement (umm)')
    _axs[2].grid(True, alpha=0.3)
    _axs[2].legend(loc='upper right')

    # Displacement residuals
    _axs[3].plot(_time_ms, _d_residual, label='Displacement Residual', color='purple')
    _axs[3].axhline(y=0, color='black', linestyle=':', alpha=0.5)
    _axs[3].set_title(f'Displacement Residual (MSE: {_d_mse:.2e})')
    _axs[3].set_ylabel('Residual (umm)')
    _axs[3].grid(True, alpha=0.3)
    _axs[3].legend(loc='upper right')

    # Signal channels colored by gradient magnitude
    _line = None
    for _j in range(_num_channels):
        _signal = signals[_i, _j, :]
        _grad_mag = abs_gradients[_i, _j, :]
        _points = np.array([_time_ms, _signal]).T.reshape(-1, 1, 2)
        _segments = np.concatenate([_points[:-1], _points[1:]], axis=1)
        _colors = _grad_mag[:-1]

        _lc = LineCollection(
            _segments, cmap='hot', norm=plt.Normalize(vmin=grad_vmin, vmax=grad_vmax)
        )
        _lc.set_array(_colors)
        _lc.set_linewidth(2)
        _line = _axs[_j + 4].add_collection(_lc)
        _axs[_j + 4].set_xlim(_time_ms[0], _time_ms[-1])
        _axs[_j + 4].set_ylim(_signal.min() * 1.1, _signal.max() * 1.1)
        _axs[_j + 4].set_title(
            f'Input Signal - {_chan_names[_j]} (colored by |doutput/dinput|)'
        )
        _axs[_j + 4].set_ylabel('Amplitude')
        _axs[_j + 4].grid(True, alpha=0.3)

    _axs[-1].set_xlabel('Time (ms)')
    _fig.suptitle(
        f'Sample {_i + 1}\n'
        f'Batch MSE: Velocity={batch_velocity_mse:.2e}, '
        f'Displacement={batch_displacement_mse:.2e}'
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # Colorbar spanning signal subplots
    if _line is not None:
        _first_bbox = _axs[4].get_position()
        _last_bbox = _axs[-1].get_position()
        _cbar_ax = _fig.add_axes(
            [0.92, _last_bbox.y0, 0.02, _first_bbox.y1 - _last_bbox.y0]
        )
        _fig.colorbar(_line, cax=_cbar_ax, label='Gradient Magnitude |doutput/dinput|')

    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ---
    *Gradient visualization shows |doutput/dinput| magnitude, indicating which input
    time points most strongly influence the model's predictions.*
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## Batch Statistics
    """)


@app.cell
def _(displacement_hat, displacement_target, np, plt, velocity_hat, velocity_target):
    _v_rmse = np.sqrt(np.mean((velocity_hat - velocity_target) ** 2, axis=1))
    _d_rmse = np.sqrt(np.mean((displacement_hat - displacement_target) ** 2, axis=1))

    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4))

    _axs[0].hist(_v_rmse, bins=50, edgecolor='black', color='#1b9e77')
    _axs[0].axvline(
        _v_rmse.mean(),
        color='#d95f02',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {_v_rmse.mean():.2f}',
    )
    _axs[0].set_title(f'Velocity RMSE (std: {_v_rmse.std():.2f})')
    _axs[0].set_xlabel('RMSE (umm/s)')
    _axs[0].set_ylabel('Count')
    _axs[0].legend()
    _axs[0].grid(True, alpha=0.3)

    _axs[1].hist(_d_rmse, bins=50, edgecolor='black', color='#d95f02')
    _axs[1].axvline(
        _d_rmse.mean(),
        color='#1b9e77',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {_d_rmse.mean():.3f}',
    )
    _axs[1].set_title(f'Displacement RMSE (std: {_d_rmse.std():.3f})')
    _axs[1].set_xlabel('RMSE (umm)')
    _axs[1].set_ylabel('Count')
    _axs[1].legend()
    _axs[1].grid(True, alpha=0.3)

    _fig.suptitle(f'Per-Sample RMSE Distribution Over Batch (N={len(_v_rmse)})')
    plt.tight_layout()
    _fig


@app.cell
def _(
    LogNorm,
    displacement_hat,
    displacement_target,
    np,
    plt,
    velocity_hat,
    velocity_target,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity correlation
    _vt = velocity_target.ravel()
    _vh = velocity_hat.ravel()
    _v_lim = max(np.abs(_vt).max(), np.abs(_vh).max()) * 1.05
    _, _, _, _im0 = _axs[0].hist2d(
        _vt,
        _vh,
        bins=100,
        norm=LogNorm(),
        cmap='viridis',
        range=[[-_v_lim, _v_lim], [-_v_lim, _v_lim]],
    )
    _axs[0].plot([-_v_lim, _v_lim], [-_v_lim, _v_lim], 'r--', alpha=0.7, label='y = x')
    _axs[0].set_xlabel('Target Velocity (umm/s)')
    _axs[0].set_ylabel('Predicted Velocity (umm/s)')
    _axs[0].set_title('Velocity Correlation')
    _axs[0].set_aspect('equal')
    _axs[0].legend()
    _fig.colorbar(_im0, ax=_axs[0], label='Count')

    # Displacement correlation
    _dt = displacement_target.ravel()
    _dh = displacement_hat.ravel()
    _d_lim = max(np.abs(_dt).max(), np.abs(_dh).max()) * 1.05
    _, _, _, _im1 = _axs[1].hist2d(
        _dt,
        _dh,
        bins=100,
        norm=LogNorm(),
        cmap='viridis',
        range=[[-_d_lim, _d_lim], [-_d_lim, _d_lim]],
    )
    _axs[1].plot([-_d_lim, _d_lim], [-_d_lim, _d_lim], 'r--', alpha=0.7, label='y = x')
    _axs[1].set_xlabel('Target Displacement (umm)')
    _axs[1].set_ylabel('Predicted Displacement (umm)')
    _axs[1].set_title('Displacement Correlation')
    _axs[1].set_aspect('equal')
    _axs[1].legend()
    _fig.colorbar(_im1, ax=_axs[1], label='Count')

    _fig.suptitle('Predicted vs Target (All Samples, All Timesteps)')
    plt.tight_layout()
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    # Summary Test Statistics, March 2026
    SCNN vs TCN - SCNN with lookahead performs better, narrows gap of free-space to fiber displacement fidelity
    """)


@app.cell
def _():
    # Data from WandB, quantities are test loss (MSE): [mean, std, min, max]
    # kernel_size = 5, dilation_base = 3; mean across dropout [0.0,0.5], displacement_loss_weight [0, 0.1]
    v_scn_mmfiber = [0.0046643, 0.0005852, 0.003954, 0.0053863]
    v_scn_free = [0.002229, 0.00048251, 0.0015218, 0.0026027]
    v_tcn_mmfiber = [0.022642, 0.0033926, 0.019913, 0.027433]
    v_tcn_free = [0.004532, 0.0010715, 0.0037145, 0.0060225]
    d_scn_mmfiber = [0.020922, 0.0014654, 0.018855, 0.02231]
    d_scn_free = [0.007327, 0.000071754, 0.0067552, 0.0082996]
    d_tcn_mmfiber = [0.051138, 0.0084596, 0.041487, 0.060263]
    d_tcn_free = [0.010315, 0.0014923, 0.0083108, 0.011818]
    return (
        d_scn_free,
        d_scn_mmfiber,
        d_tcn_free,
        d_tcn_mmfiber,
        v_scn_free,
        v_scn_mmfiber,
        v_tcn_free,
        v_tcn_mmfiber,
    )


@app.cell
def _(
    d_scn_free,
    d_scn_mmfiber,
    d_tcn_free,
    d_tcn_mmfiber,
    np,
    plt,
    v_scn_free,
    v_scn_mmfiber,
    v_tcn_free,
    v_tcn_mmfiber,
):
    def to_rmse(mse_stats):
        """Convert [mean, std, min, max] MSE stats to RMSE."""
        mean, std, mn, mx = mse_stats
        return np.sqrt(mean), std * 1 / (2 * np.sqrt(mean)), np.sqrt(mn), np.sqrt(mx)

    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))

    for _ax, _title, _data in [
        (
            _axs[0],
            'Velocity RMSE',
            [
                ('SCNN Fiber', v_scn_mmfiber),
                ('SCNN Free Space', v_scn_free),
                ('TCN Fiber', v_tcn_mmfiber),
                ('TCN Free Space', v_tcn_free),
            ],
        ),
        (
            _axs[1],
            'Displacement RMSE',
            [
                ('SCNN Fiber', d_scn_mmfiber),
                ('SCNN Free Space', d_scn_free),
                ('TCN Fiber', d_tcn_mmfiber),
                ('TCN Free Space', d_tcn_free),
            ],
        ),
    ]:
        _labels = [d[0] for d in _data]
        _means = [to_rmse(d[1])[0] for d in _data]
        _stds = [to_rmse(d[1])[1] for d in _data]
        _mins = [to_rmse(d[1])[2] for d in _data]
        _maxs = [to_rmse(d[1])[3] for d in _data]
        _x = np.arange(len(_labels))
        _colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

        _bars = _ax.bar(
            _x,
            _means,
            yerr=_stds,
            capsize=5,
            color=_colors,
            edgecolor='black',
            zorder=3,
        )
        for _k in range(len(_x)):
            _ax.hlines(
                _mins[_k],
                _x[_k] - 0.3,
                _x[_k] + 0.3,
                colors='#140de6',
                linewidths=2,
                zorder=4,
            )
            _ax.hlines(
                _maxs[_k],
                _x[_k] - 0.3,
                _x[_k] + 0.3,
                colors='#140de6',
                linewidths=2,
                zorder=4,
            )
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_labels, rotation=30, ha='right')
        _ax.set_title(_title)
        _unit = '(umm/ms)' if 'Velocity' in _title else '(umm)'
        _ax.set_ylabel(f'RMSE {_unit}')
        _ax.grid(True, alpha=0.3, axis='y')

    _fig.suptitle('Test RMSE by Model and Experiment Mode')
    plt.tight_layout()
    _fig


@app.cell
def _(
    d_scn_free,
    d_scn_mmfiber,
    d_tcn_free,
    d_tcn_mmfiber,
    np,
    plt,
    v_scn_free,
    v_scn_mmfiber,
    v_tcn_free,
    v_tcn_mmfiber,
):
    def _ratio_rmse(mse_mmfiber, mse_free):
        """Compute RMSE(mmfiber) / RMSE(free) for [mean, std, min, max]."""
        _rm = np.sqrt(mse_mmfiber[0])
        _rf = np.sqrt(mse_free[0])
        mean_ratio = _rm / _rf
        # Propagate std via delta method: sigma_RMSE = sigma_MSE / (2*sqrt(mean_MSE))
        _sm = mse_mmfiber[1] / (2 * _rm)
        _sf = mse_free[1] / (2 * _rf)
        # Then ratio error propagation: sigma_ratio = ratio * sqrt((sigma_a/a)^2 + (sigma_b/b)^2)
        std_ratio = mean_ratio * np.sqrt((_sm / _rm) ** 2 + (_sf / _rf) ** 2)
        min_ratio = np.sqrt(mse_mmfiber[2]) / np.sqrt(mse_free[3])
        max_ratio = np.sqrt(mse_mmfiber[3]) / np.sqrt(mse_free[2])
        return [mean_ratio, std_ratio, min_ratio, max_ratio]

    _v_scn_ratio = _ratio_rmse(v_scn_mmfiber, v_scn_free)
    _v_tcn_ratio = _ratio_rmse(v_tcn_mmfiber, v_tcn_free)
    _d_scn_ratio = _ratio_rmse(d_scn_mmfiber, d_scn_free)
    _d_tcn_ratio = _ratio_rmse(d_tcn_mmfiber, d_tcn_free)

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))

    for _ax, _title, _ratios in [
        (
            _axs[0],
            'Velocity RMSE Ratio',
            [('SCNN', _v_scn_ratio), ('TCN', _v_tcn_ratio)],
        ),
        (
            _axs[1],
            'Displacement RMSE Ratio',
            [('SCNN', _d_scn_ratio), ('TCN', _d_tcn_ratio)],
        ),
    ]:
        _labels = [r[0] for r in _ratios]
        _means = [r[1][0] for r in _ratios]
        _stds = [r[1][1] for r in _ratios]
        _mins = [r[1][2] for r in _ratios]
        _maxs = [r[1][3] for r in _ratios]
        _x = np.arange(len(_labels))
        _colors = ['#1b9e77', '#d95f02']

        _ax.bar(
            _x,
            _means,
            yerr=_stds,
            capsize=5,
            color=_colors,
            edgecolor='black',
            zorder=3,
        )
        for _k in range(len(_x)):
            _ax.hlines(
                _mins[_k],
                _x[_k] - 0.3,
                _x[_k] + 0.3,
                colors='#140de6',
                linewidths=2,
                zorder=4,
            )
            _ax.hlines(
                _maxs[_k],
                _x[_k] - 0.3,
                _x[_k] + 0.3,
                colors='#140de6',
                linewidths=2,
                zorder=4,
            )
        _ax.axhline(y=1, color='black', linestyle=':', alpha=0.5)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_labels)
        _ax.set_title(_title)
        # _ax.set_ylabel('RMSE(MM Fiber) / RMSE(Free Space)')
        _ax.grid(True, alpha=0.3, axis='y')

    _fig.suptitle('Fiber / Free Space RMSE Ratio by Model')
    plt.tight_layout()
    _fig


@app.cell
def _():
    return


if __name__ == '__main__':
    app.run()
