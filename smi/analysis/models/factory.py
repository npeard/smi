"""Model factory for creating neural network models."""

import logging
from typing import Any

from torch import nn

from .barland_cnn import BarlandCNN, BarlandCNNConfig
from .lstm import LSTM, LSTMConfig
from .mamba import Mamba, MambaConfig
from .mlp import MLP, MLPConfig
from .scnn import SCNN, SCNNConfig
from .tcan import TCAN, TCANConfig
from .tcn import TCN, TCNConfig

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
