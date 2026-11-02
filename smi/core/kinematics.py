"""Velocity <-> displacement conversion in the project's stored units
(um and um/ms).
"""

import numpy as np
import torch

# Milliseconds per second. Sample rates enter in Hz, so a time derivative is
# naturally per-second; dividing by this converts to per-ms, so velocity * dt
# comes out in um.
MS_PER_S = 1.0e3


def integrate_velocity(
    velocity_waveform: np.ndarray | torch.Tensor, sample_rate: float
) -> np.ndarray | torch.Tensor:
    """Integrate velocity waveform to get displacement waveform using
    cumulative integration.

    Compatible with both NumPy arrays and PyTorch tensors.

    Args:
        velocity_waveform: Velocity waveform (microns/ms)
                          Can be a NumPy array or PyTorch tensor
                          For PyTorch tensors, supports batch dimensions
                          [batch_size, channels, signal_length]
        sample_rate: Sample rate of the velocity waveform (Hz)

    Returns:
        Displacement waveform (microns) in the same format as input
    """
    # Check if input is a PyTorch tensor
    # More robust check using module name instead of attribute
    is_torch = 'torch' in str(type(velocity_waveform).__module__)
    # Time step in milliseconds, because velocity is um/ms: the product
    # velocity * dt is then um, matching the displacement unit.
    dt = MS_PER_S / sample_rate

    if is_torch:
        # Handle batch dimensions if present
        if len(velocity_waveform.shape) > 1:
            _batch_size, _signal_length = velocity_waveform.shape

            displacement = (
                torch.cumsum(velocity_waveform, dim=-1) * dt
            )  # narrowed to Tensor at runtime via is_torch
            # Shift displacement to start at zero
            displacement = displacement - displacement[:, 0].unsqueeze(-1)

        else:
            # Single waveform case
            displacement = (
                torch.cumsum(velocity_waveform, dim=0) * dt
            )  # narrowed to Tensor at runtime via is_torch

            # Shift displacement to start at zero
            displacement = displacement - displacement[0]

        return displacement

    else:
        # NumPy implementation (original)
        # Simple cumulative integration (cumulative sum * dt)
        displacement = np.cumsum(velocity_waveform) * dt

        # Shift displacement to start at zero at the beginning of the trace
        return displacement - displacement[0]


def derivative_displacement(
    displacement_waveform: np.ndarray | torch.Tensor, sample_rate: float
) -> np.ndarray | torch.Tensor:
    """Calculate the time derivative of a displacement waveform to get velocity.

    This method uses central differences to compute the derivative.
    Compatible with both NumPy arrays and PyTorch tensors.

    Args:
        displacement_waveform: Displacement waveform (microns)
                              Can be a NumPy array or PyTorch tensor
                              For PyTorch tensors, supports batch dimensions
                              [batch_size, signal_length]
        sample_rate: Sample rate of the displacement waveform (Hz)

    Returns:
        Velocity waveform (microns/ms) in the same format as input
    """
    # Check if input is a PyTorch tensor
    is_torch = 'torch' in str(type(displacement_waveform).__module__)

    # Time step in milliseconds, so d(um)/dt comes out as um/ms rather
    # than um/s -- see the module docstring's units note.
    dt = MS_PER_S / sample_rate

    if is_torch:
        # Handle batch dimensions if present
        if len(displacement_waveform.shape) > 1:
            _batch_size, _signal_length = displacement_waveform.shape
            velocity = torch.zeros_like(displacement_waveform)

            # First point (forward difference)
            velocity[:, 0] = (
                displacement_waveform[:, 1] - displacement_waveform[:, 0]
            ) / dt

            # Middle points (central difference)
            velocity[:, 1:-1] = (
                displacement_waveform[:, 2:] - displacement_waveform[:, :-2]
            ) / (2 * dt)

            # Last point (backward difference)
            velocity[:, -1] = (
                displacement_waveform[:, -1] - displacement_waveform[:, -2]
            ) / dt
        else:
            velocity = torch.zeros_like(displacement_waveform)

            # First point (forward difference)
            velocity[0] = (displacement_waveform[1] - displacement_waveform[0]) / dt

            # Middle points (central difference)
            velocity[1:-1] = (
                displacement_waveform[2:] - displacement_waveform[:-2]
            ) / (2 * dt)

            # Last point (backward difference)
            velocity[-1] = (displacement_waveform[-1] - displacement_waveform[-2]) / dt
    else:
        # NumPy implementation
        velocity = np.zeros_like(displacement_waveform)

        # First point (forward difference)
        velocity[0] = (displacement_waveform[1] - displacement_waveform[0]) / dt

        # Middle points (central difference)
        velocity[1:-1] = (displacement_waveform[2:] - displacement_waveform[:-2]) / (
            2 * dt
        )

        # Last point (backward difference)
        velocity[-1] = (displacement_waveform[-1] - displacement_waveform[-2]) / dt

    return velocity
