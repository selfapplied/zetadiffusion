"""
tensor.py

Effective Scaling Tensor D_ij and related manifold operations.
"""

import numpy as np
from .renorm import DELTA_F

def effective_scaling_tensor(metric_g_ij, hurst_field, kernel_matrix, twist_coupling=0.0):
    """
    Computes the Complex Effective Scaling Tensor D_ij(x).
    
    D_ij = delta_F * g_ij + Integral(K(x,y) * H(y) dy)
    
    Generalized to complex: D_ij = Re[D_ij] + i * Im[D_ij]
    
    Args:
        metric_g_ij: Base metric tensor (can be scalar, 1D array, or 2D tensor).
        hurst_field: The memory/persistence field H(y).
        kernel_matrix: Interaction kernel K(x,y).
        twist_coupling: Coefficient for the imaginary (rotational) component.
        
    Returns:
        Complex-valued scaling tensor D_ij.
    """
    # Interaction term: Integral(K(x,y) * H(y) dy) approximated as matrix-vector product
    interaction = np.dot(kernel_matrix, hurst_field)
    
    # Handle metric shape
    metric = np.asarray(metric_g_ij)
    if metric.ndim == 0:
        # Scalar metric -> identity scaling
        metric = np.ones_like(interaction)
    elif metric.ndim == 1:
        # 1D metric field
        pass
    else:
        # 2D tensor -> extract diagonal or use trace
        if metric.shape[0] == metric.shape[1]:
            metric = np.trace(metric, axis1=0, axis2=1) if metric.ndim > 2 else np.diag(metric)
        else:
            metric = np.ones_like(interaction)
    
    # Real part: D_ij = delta_F * g_ij + interaction
    real_part = DELTA_F * metric + interaction
    
    # Imaginary part: Twist (vorticity) from memory coupling
    imag_part = twist_coupling * interaction
    
    # Construct complex tensor
    D_complex = real_part + 1j * imag_part
    
    return D_complex
