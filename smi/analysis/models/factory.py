"""Model factory for creating neural network models."""

import logging
from typing import Any

from torch import nn

from .barland_cnn import BarlandCNN, BarlandCNNConfig
from .lstm import LSTM, LSTMConfig
from .mamba import Mamba, MambaConfig
from .mlp import MLP, MLPConfig
from .neural_operator import NeuralOperator, NeuralOperatorConfig
from .patch_transformer import PatchTransformer, PatchTransformerConfig
from .scnn import SCNN, SCNNConfig
from .tcan import TCAN, TCANConfig
from .tcn import TCN, TCNConfig
from .unet_transformer import UNetTransformer, UNetTransformerConfig

logger = logging.getLogger(__name__)

# Every architecture is built the same way -- config dataclass from the
# remaining hparams, then the module -- so the dispatch is a table rather than
# an if/elif chain that grows a branch per model.
MODEL_REGISTRY: dict[str, tuple[type, type[nn.Module]]] = {
    'Barland': (BarlandCNNConfig, BarlandCNN),
    'TCN': (TCNConfig, TCN),
    'SCNN': (SCNNConfig, SCNN),
    'TCAN': (TCANConfig, TCAN),
    'LSTM': (LSTMConfig, LSTM),
    'MLP': (MLPConfig, MLP),
    'Mamba': (MambaConfig, Mamba),
    'PatchTransformer': (PatchTransformerConfig, PatchTransformer),
    'UNetTransformer': (UNetTransformerConfig, UNetTransformer),
    'NeuralOperator': (NeuralOperatorConfig, NeuralOperator),
}


def create_model(model_hparams: dict[str, Any]) -> nn.Module:
    """Create model instance based on configuration.

    Args:
        model_hparams: Model hyperparameters dictionary containing 'type' key
                      and model-specific configuration parameters.

    Returns:
        A PyTorch model instance based on the configuration. Always a module:
        an unrecognised type raises rather than returning None.

    Raises:
        ValueError: If unknown model type is specified.
        ImportError: If required dependencies are not available.
    """
    model_type = model_hparams['type']
    # Remove 'type' from params since it's not part of any dataclass
    params = model_hparams.copy()
    params.pop('type', None)
    params.pop('total_params', None)  # Remove total_params if present
    params.pop('receptive_field', None)  # Remove receptive_field if present

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model type: {model_type}')

    config_cls, model_cls = MODEL_REGISTRY[model_type]
    logger.debug(f'Creating {model_cls.__name__} model...')
    return model_cls(config_cls(**params))


#: Model types that see samples after the timestamp they predict.
#:
#: This is not "SCNN only". ``windowize`` labels each window at its centre and
#: the windowed torch models average one prediction across the whole span, so
#: Barland, MLP, LightGBM and Ridge all use lookahead that the TCN is denied.
#: Any table comparing them to the analytic baseline has to say so. LightGBM
#: and Ridge are tabular baselines outside MODEL_REGISTRY.
_NONCAUSAL_TYPES = frozenset({'SCNN', 'Barland', 'MLP', 'LightGBM', 'Ridge'})
#: Model types built in both directions, chosen by their ``causal`` field.
_CAUSAL_FLAG_TYPES = frozenset(
    {'PatchTransformer', 'UNetTransformer', 'NeuralOperator'}
)


def is_causal(model_hparams: dict[str, Any]) -> bool:
    """Whether a model uses only past and present samples.

    The one place this is decided, so a results record cannot disagree with
    the sweep that produced it. ``tests/test_model_causality.py`` checks it
    against each registry model's measured behaviour.

    Args:
        model_hparams: Hyperparameters with a ``type`` key; an LSTM also needs
            its ``bidirectional`` flag, a type in ``_CAUSAL_FLAG_TYPES`` its
            ``causal`` flag.

    Returns:
        True if the model is causal.
    """
    if model_hparams['type'] in _NONCAUSAL_TYPES:
        return False
    if model_hparams['type'] in _CAUSAL_FLAG_TYPES:
        return model_hparams['causal']
    if model_hparams['type'] == 'LSTM':
        return not model_hparams['bidirectional']
    # The remaining registry types (TCN, TCAN, Mamba, ...) are causal by
    # construction, with no direction field to read.
    return True
