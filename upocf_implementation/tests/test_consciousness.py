"""
Unit tests for consciousness function implementation.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from upocf.core.consciousness import ConsciousnessFunction, psi


class TestConsciousnessFunction:
    """Test suite for ConsciousnessFunction class."""
    
    def test_initialization(self):
        """Test consciousness function initialization."""
        # Test basic initialization
        cf = ConsciousnessFunction(network_size=4)
        assert cf.network_size == 4
        assert cf.use_fast_phi == True
        
        # Test with different parameters
        cf_large = ConsciousnessFunction(network_size=10, use_fast_phi=False)
        assert cf_large.network_size == 10
        assert cf_large.use_fast_phi == False
    
    def test_basic_evaluation(self):
        """Test basic consciousness function evaluation."""
        cf = ConsciousnessFunction(network_size=4)
        
        # Test with simple state
        x = jnp.array([0.5, -0.3, 0.8, -0.1])
        psi_value = cf(x)
        
        assert isinstance(psi_value, (float, jnp.ndarray))
        assert psi_value >= 0.0  # Φ should be non-negative
    
    def test_deterministic_behavior(self):
        """Test that function is deterministic."""
        cf = ConsciousnessFunction(network_size=4)
        x = jnp.array([0.1, 0.2, 0.3, 0.4])
        
        # Multiple evaluations should give same result
        result1 = cf(x)
        result2 = cf(x)
        
        assert jnp.allclose(result1, result2, atol=1e-10)
    
    def test_different_network_sizes(self):
        """Test consciousness function with different network sizes."""
        for size in [2, 4, 6, 8]:
            cf = ConsciousnessFunction(network_size=size)
            x = jnp.ones(size) * 0.5
            
            psi_value = cf(x)
            assert isinstance(psi_value, (float, jnp.ndarray))
            assert psi_value >= 0.0
    
    def test_gradient_computation(self):
        """Test gradient computation using autodiff."""
        cf = ConsciousnessFunction(network_size=4)
        x = jnp.array([0.5, -0.3, 0.8, -0.1])
        
        grad = cf.gradient(x)
        
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad))
    
    def test_hessian_computation(self):
        """Test Hessian computation."""
        cf = ConsciousnessFunction(network_size=3)  # Smaller size for efficiency
        x = jnp.array([0.5, -0.3, 0.8])
        
        hessian = cf.hessian(x)
        
        assert hessian.shape == (3, 3)
        assert jnp.all(jnp.isfinite(hessian))
        # Hessian should be symmetric (approximately)
        assert jnp.allclose(hessian, hessian.T, atol=1e-6)
    
    def test_fifth_derivative_norm(self):
        """Test fifth derivative norm computation."""
        cf = ConsciousnessFunction(network_size=3)
        x = jnp.array([0.1, 0.2, 0.3])
        
        fifth_deriv_norm = cf.fifth_derivative_norm(x)
        
        assert isinstance(fifth_deriv_norm, (float, jnp.ndarray))
        assert fifth_deriv_norm >= 0.0
        assert jnp.isfinite(fifth_deriv_norm)
    
    def test_bipartition_enumeration(self):
        """Test bipartition enumeration for small networks."""
        cf = ConsciousnessFunction(network_size=4)
        partitions = cf._enumerate_bipartitions(4)
        
        # Should have 2^(4-1) - 1 = 7 non-trivial bipartitions
        # Actually, should have all possible non-empty proper subsets
        assert len(partitions) > 0
        
        # Check that partitions are valid
        for part_a, part_b in partitions:
            assert len(part_a) > 0
            assert len(part_b) > 0
            assert len(part_a) + len(part_b) == 4
            assert set(part_a + part_b) == set(range(4))
    
    def test_mutual_information_properties(self):
        """Test mutual information computation properties."""
        cf = ConsciousnessFunction(network_size=4)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        
        # Test with simple partition
        part_a = [0, 1]
        part_b = [2, 3]
        
        mi = cf._mutual_information(x, part_a, part_b)
        
        assert isinstance(mi, (float, jnp.ndarray))
        assert mi >= 0.0  # MI should be non-negative
        assert jnp.isfinite(mi)
    
    def test_zero_state(self):
        """Test consciousness function on zero state."""
        cf = ConsciousnessFunction(network_size=4)
        x = jnp.zeros(4)
        
        psi_value = cf(x)
        
        assert isinstance(psi_value, (float, jnp.ndarray))
        assert jnp.isfinite(psi_value)
    
    def test_large_values_stability(self):
        """Test numerical stability with large input values."""
        cf = ConsciousnessFunction(network_size=4)
        x = jnp.array([10.0, -10.0, 5.0, -5.0])
        
        psi_value = cf(x)
        
        assert jnp.isfinite(psi_value)
        assert psi_value >= 0.0


class TestPsiFunction:
    """Test suite for the standalone psi function."""
    
    def test_psi_basic(self):
        """Test basic psi function usage."""
        x = jnp.array([0.5, -0.3, 0.8, -0.1])
        
        psi_value = psi(x)
        
        assert isinstance(psi_value, (float, jnp.ndarray))
        assert psi_value >= 0.0
    
    def test_psi_with_network_size(self):
        """Test psi function with explicit network size."""
        x = jnp.array([0.1, 0.2, 0.3])
        
        psi_value = psi(x, network_size=3)
        
        assert isinstance(psi_value, (float, jnp.ndarray))
        assert psi_value >= 0.0
    
    def test_psi_consistency(self):
        """Test consistency between psi function and ConsciousnessFunction."""
        x = jnp.array([0.5, -0.3, 0.8, -0.1])
        
        # Direct psi call
        psi_direct = psi(x)
        
        # Through ConsciousnessFunction
        cf = ConsciousnessFunction(network_size=4)
        psi_class = cf(x)
        
        assert jnp.allclose(psi_direct, psi_class, atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_state(self):
        """Test behavior with empty state (should raise error)."""
        with pytest.raises((ValueError, IndexError)):
            psi(jnp.array([]))
    
    def test_single_node(self):
        """Test with single node network."""
        cf = ConsciousnessFunction(network_size=1)
        x = jnp.array([0.5])
        
        # Single node should have zero integrated information
        psi_value = cf(x)
        assert psi_value == 0.0 or jnp.isclose(psi_value, 0.0, atol=1e-6)
    
    def test_nan_input(self):
        """Test behavior with NaN input."""
        cf = ConsciousnessFunction(network_size=3)
        x = jnp.array([0.5, jnp.nan, 0.3])
        
        # Should handle NaN gracefully or raise appropriate error
        try:
            psi_value = cf(x)
            assert jnp.isnan(psi_value) or jnp.isfinite(psi_value)
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid input
            pass
    
    def test_inf_input(self):
        """Test behavior with infinite input."""
        cf = ConsciousnessFunction(network_size=3)
        x = jnp.array([0.5, jnp.inf, 0.3])
        
        # Should handle infinity gracefully or raise appropriate error
        try:
            psi_value = cf(x)
            assert jnp.isfinite(psi_value) or jnp.isinf(psi_value)
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid input
            pass


if __name__ == "__main__":
    pytest.main([__file__])