import marimo

__generated_with = '0.20.4'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    from torchview import draw_graph

    from smi.learning.models.factory import create_model

    return create_model, draw_graph, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Model Architecture Viewer

    Select a model config to visualize its architecture as a block diagram.
    """)


@app.cell
def _():
    from smi.learning.configs import model_config

    config = model_config('tcn-config')
    return (config,)


@app.cell
def _(config, create_model):
    _model_hparams = config.model_config

    # Handle list-valued hyperparams (from sweep configs) by taking first element
    _clean_hparams = {}
    for k, v in _model_hparams.items():
        if (isinstance(v, list) and k != 'temporal_channels') or (
            k == 'temporal_channels' and isinstance(v, list) and isinstance(v[0], list)
        ):
            _clean_hparams[k] = v[0]
        else:
            _clean_hparams[k] = v

    model = create_model(_clean_hparams)
    in_channels = _clean_hparams.get('in_channels', 3)
    sequence_length = _clean_hparams.get('sequence_length', 16384)
    return in_channels, model, sequence_length


@app.cell
def _(mo, model):
    _total_params = getattr(model, 'total_params', None)
    _receptive_field = getattr(model, 'receptive_field', None)

    _lines = [f'**Model type:** `{type(model).__name__}`']
    if _total_params is not None:
        _lines.append(f'**Total parameters:** {int(_total_params):,}')
    if _receptive_field is not None:
        _lines.append(f'**Receptive field:** {int(_receptive_field):,} samples')

    mo.md('\n\n'.join(_lines))


@app.cell
def _(draw_graph, in_channels, model, sequence_length):
    graph = draw_graph(
        model, input_size=(1, in_channels, sequence_length), expand_nested=True, depth=3
    )
    graph.visual_graph


if __name__ == '__main__':
    app.run()
