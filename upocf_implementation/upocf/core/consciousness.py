"""
Core consciousness function implementation.

Defines Ψ(x) := Φ_IIT(x) with automatic differentiation support.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np
from typing import Tuple, Optional
import pyphi

# Enable double precision for numerical stability
jax.config.update("jax_enable_x64", True)


class ConsciousnessFunction:
    """
    Operational definition of consciousness function Ψ(x) := Φ_IIT(x).
    
    Args:
        network_size: Number of nodes in the network
        use_fast_phi: If True, use simplified Φ calculation for toy systems
    """
    
    def __init__(self, network_size: int, use_fast_phi: bool = True):
        self.network_size = network_size
        self.use_fast_phi = use_fast_phi
        
        # Precompute bipartitions for efficiency
        if use_fast_phi and network_size <= 8:
            self.bipartitions = self._enumerate_bipartitions(network_size)
    
    def __call__(self, x: jnp.ndarray) -> float:
        """
        Compute consciousness function Ψ(x) = Φ_IIT(x).
        
        Args:
            x: Network state vector of shape (n,) where n = network_size
            
        Returns:
            Φ value representing integrated information
        """
        if self.use_fast_phi and self.network_size <= 8:
            return self._fast_phi(x)
        else:
            return self._pyphi_phi(x)
    
    def _fast_phi(self, x: jnp.ndarray) -> float:
        """
        Fast Φ calculation for toy systems (≤8 nodes).
        
        Φ = max_P min{I(P)} over all bipartitions P.
        """
        phi_values = []
        
        for partition in self.bipartitions:
            part_a, part_b = partition
            mi = self._mutual_information(x, part_a, part_b)
            phi_values.append(mi)
        
        return jnp.max(jnp.array(phi_values)) if phi_values else 0.0
    
    def _mutual_information(self, x: jnp.ndarray, part_a: list, part_b: list) -> float:
        """
        Compute mutual information across a bipartition.
        
        For continuous states, use Gaussian approximation:
        I(A;B) = 0.5 * log(det(Σ_A) * det(Σ_B) / det(Σ_AB))
        """
        # Extract subvectors
        x_a = x[part_a]
        x_b = x[part_b]
        x_ab = jnp.concatenate([x_a, x_b])
        
        # Gaussian MI approximation (simplified for demo)
        # In practice, would use proper covariance estimation
        var_a = jnp.var(x_a) + 1e-8  # Regularization
        var_b = jnp.var(x_b) + 1e-8
        var_ab = jnp.var(x_ab) + 1e-8
        
        # Simplified MI estimate
        mi = 0.5 * jnp.log(var_a * var_b / var_ab)
        return jnp.maximum(mi, 0.0)  # Ensure non-negative
    
    def _pyphi_phi(self, x: jnp.ndarray) -> float:
        """
        Full PyPhi-based Φ calculation for larger systems.
        
        Note: This is a placeholder - actual implementation would require
        converting continuous x to discrete states for PyPhi.
        """
        # Convert continuous state to binary for PyPhi
        binary_state = (x > jnp.median(x)).astype(int)
        
        # Create PyPhi network (simplified example)
        # In practice, would need proper connectivity matrix
        network = pyphi.Network(
            tpm=self._create_tpm(binary_state),
            connectivity_matrix=jnp.eye(self.network_size)
        )
        
        subsystem = pyphi.Subsystem(network, binary_state, range(self.network_size))
        phi = pyphi.compute.phi(subsystem)
        
        return float(phi)
    
    def _create_tpm(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Create transition probability matrix from state.
        Simplified implementation for demo purposes.
        """
        n = len(state)
        tpm = jnp.zeros((2**n, n))
        # Placeholder: identity mapping
        for i in range(2**n):
            binary_i = jnp.array([(i >> j) & 1 for j in range(n)])
            tpm = tpm.at[i].set(binary_i)
        return tpm
    
    def _enumerate_bipartitions(self, n: int) -> list:
        """
        Enumerate all non-trivial bipartitions of n nodes.
        
        Returns list of (part_a, part_b) tuples.
        """
        from itertools import combinations
        
        partitions = []
        nodes = list(range(n))
        
        # Generate all possible subset sizes (excluding empty and full set)
        for size_a in range(1, n):
            for part_a in combinations(nodes, size_a):
                part_b = [node for node in nodes if node not in part_a]
                if len(part_b) > 0:  # Ensure non-empty complement
                    partitions.append((list(part_a), part_b))
        
        return partitions

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute ∇Ψ using automatic differentiation."""
        return grad(self.__call__)(x)
    
    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute ∇²Ψ using automatic differentiation."""
        return jacfwd(grad(self.__call__))(x)
    
    def fifth_derivative_norm(self, x: jnp.ndarray) -> float:
        """
        Compute ||Ψ⁽⁵⁾|| for empirical bound studies.
        
        Uses finite differences for 5th derivative approximation.
        """
        h = 1e-3  # Step size
        n = len(x)
        
        # Compute 5th derivative using finite differences
        # This is computationally expensive but needed for bound validation
        fifth_derivatives = []
        
        for i in range(n):
            # Create perturbation vector
            e_i = jnp.zeros_like(x)
            e_i = e_i.at[i].set(1.0)
            
            # 5-point stencil for 5th derivative
            f_minus2 = self(x - 2*h*e_i)
            f_minus1 = self(x - h*e_i)
            f_0 = self(x)
            f_plus1 = self(x + h*e_i)
            f_plus2 = self(x + 2*h*e_i)
            
            # 5th derivative approximation (simplified)
            d5f = (f_plus2 - 4*f_plus1 + 6*f_0 - 4*f_minus1 + f_minus2) / h**5
            fifth_derivatives.append(d5f)
        
        return jnp.linalg.norm(jnp.array(fifth_derivatives))


# Convenience function for direct usage
def psi(x: jnp.ndarray, network_size: Optional[int] = None) -> float:
    """
    Direct consciousness function evaluation.
    
    Args:
        x: Network state vector
        network_size: Size of network (inferred from x if None)
        
    Returns:
        Ψ(x) = Φ_IIT(x) value
    """
    if network_size is None:
        network_size = len(x)
    
    consciousness_fn = ConsciousnessFunction(network_size)
    return consciousness_fn(x)


# Export main interface
__all__ = ['ConsciousnessFunction', 'psi']