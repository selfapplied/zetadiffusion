#!.venv/bin/python
"""
Interactive Computational Demonstration

Visualizes:
1. Pascal's triangle animation with sin/cos path splitting
2. tan(θ) rotation on unit circle synchronized with binomial coefficients
3. Transformer attention weights mapped to angular projections
4. Real-time Nash equilibrium solver for Guardian-Generator game
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import json

class PascalTriangleVisualizer:
    """Animate Pascal's triangle with sin/cos path splitting."""
    
    def __init__(self, n_rows: int = 10):
        self.n_rows = n_rows
        self.triangle = self._build_triangle()
    
    def _build_triangle(self) -> np.ndarray:
        """Build Pascal's triangle."""
        triangle = np.zeros((self.n_rows, self.n_rows), dtype=int)
        
        for i in range(self.n_rows):
            for j in range(i + 1):
                if j == 0 or j == i:
                    triangle[i, j] = 1
                else:
                    triangle[i, j] = triangle[i-1, j-1] + triangle[i-1, j]
        
        return triangle
    
    def get_sin_cos_path(self, row: int, col: int) -> tuple:
        """
        Map binomial coefficient to sin/cos path.
        
        Path splits based on parity: even → cos, odd → sin
        """
        value = self.triangle[row, col]
        
        # Angle from position
        angle = (col / (row + 1)) * 2 * np.pi
        
        # Split based on value parity
        if value % 2 == 0:
            # Even: use cosine
            x = np.cos(angle) * (value / 100.0)  # Normalize
            y = np.sin(angle) * (value / 100.0)
        else:
            # Odd: use sine (shifted)
            x = np.sin(angle + np.pi/4) * (value / 100.0)
            y = np.cos(angle + np.pi/4) * (value / 100.0)
        
        return (x, y)
    
    def visualize(self, output_file: str = ".out/pascal_triangle.png"):
        """Create static visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw triangle
        for i in range(self.n_rows):
            for j in range(i + 1):
                value = self.triangle[i, j]
                x = j - i/2
                y = -i
                
                # Color based on value
                color = 'blue' if value % 2 == 0 else 'red'
                ax.text(x, y, str(value), ha='center', va='center',
                       fontsize=8, color=color)
        
        ax.set_xlim(-self.n_rows/2 - 1, self.n_rows/2 + 1)
        ax.set_ylim(-self.n_rows - 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Pascal's Triangle with Sin/Cos Path Splitting")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")

class TanRotationVisualizer:
    """Visualize tan(θ) rotation on unit circle synchronized with binomials."""
    
    def __init__(self, n_points: int = 100):
        self.n_points = n_points
        self.angles = np.linspace(0, 2 * np.pi, n_points)
    
    def get_binomial_sync(self, angle: float) -> float:
        """
        Synchronize binomial coefficient with angle.
        
        Use angle to index into Pascal's triangle row.
        """
        # Map angle to row index
        row = int((angle / (2 * np.pi)) * 10) % 10
        
        # Get binomial coefficient from row
        col = int((angle / (2 * np.pi)) * row) % (row + 1) if row > 0 else 0
        
        # Compute binomial
        if row == 0:
            return 1
        binom = np.math.comb(row, col) if row >= col else 0
        
        return binom
    
    def visualize(self, output_file: str = ".out/tan_rotation.png"):
        """Visualize tan rotation on unit circle."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Unit circle
        circle_x = np.cos(self.angles)
        circle_y = np.sin(self.angles)
        ax1.plot(circle_x, circle_y, 'k-', linewidth=2)
        
        # Tan vectors
        tan_values = np.tan(self.angles)
        tan_values = np.clip(tan_values, -5, 5)  # Clip for visualization
        
        for i in range(0, len(self.angles), 5):
            angle = self.angles[i]
            tan_val = tan_values[i]
            binom = self.get_binomial_sync(angle)
            
            # Vector from unit circle
            x_start = np.cos(angle)
            y_start = np.sin(angle)
            x_end = x_start + np.cos(angle) * tan_val * 0.1
            y_end = y_start + np.sin(angle) * tan_val * 0.1
            
            # Color by binomial
            color_intensity = min(binom / 100.0, 1.0)
            ax1.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                     head_width=0.05, head_length=0.05, 
                     color=plt.cm.viridis(color_intensity))
        
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal')
        ax1.set_title("tan(θ) Rotation on Unit Circle\n(Synchronized with Binomials)")
        ax1.grid(True, alpha=0.3)
        
        # Tan curve
        ax2.plot(self.angles, tan_values, 'b-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=np.pi/2, color='r', linestyle='--', alpha=0.5, label='π/2')
        ax2.axvline(x=3*np.pi/2, color='r', linestyle='--', alpha=0.5, label='3π/2')
        ax2.set_xlabel('Angle θ')
        ax2.set_ylabel('tan(θ)')
        ax2.set_title('Tangent Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")

class AttentionAngularMapper:
    """Map transformer attention weights to angular projections."""
    
    def __init__(self, n_tokens: int = 10, n_heads: int = 4):
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.attention = self._generate_attention()
    
    def _generate_attention(self) -> np.ndarray:
        """Generate synthetic attention weights."""
        # Create attention matrix with some structure
        attention = np.random.rand(self.n_heads, self.n_tokens, self.n_tokens)
        
        # Normalize rows (softmax-like)
        for h in range(self.n_heads):
            for i in range(self.n_tokens):
                attention[h, i, :] = attention[h, i, :] / np.sum(attention[h, i, :])
        
        return attention
    
    def map_to_angles(self, head: int = 0) -> np.ndarray:
        """
        Map attention weights to angular projections.
        
        Each token position maps to an angle, attention weight
        determines radius.
        """
        angles = np.linspace(0, 2 * np.pi, self.n_tokens, endpoint=False)
        
        # Get attention for this head (average over query positions)
        attn_avg = np.mean(self.attention[head, :, :], axis=0)
        
        return angles, attn_avg
    
    def visualize(self, output_file: str = ".out/attention_angular.png"):
        """Visualize attention mapped to angles."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for head in range(min(self.n_heads, 4)):
            ax = axes[head]
            
            angles, attn_weights = self.map_to_angles(head)
            
            # Polar plot
            ax = plt.subplot(2, 2, head + 1, projection='polar')
            
            # Plot attention as radius
            radii = attn_weights * 10  # Scale for visibility
            
            ax.bar(angles, radii, width=2*np.pi/self.n_tokens, 
                  alpha=0.7, color=plt.cm.viridis(attn_weights))
            
            ax.set_title(f'Head {head + 1}: Attention → Angular Projection', pad=20)
            ax.set_ylim(0, max(radii) * 1.2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")

class NashEquilibriumSolver:
    """Real-time Nash equilibrium solver for Guardian-Generator game."""
    
    def __init__(self, coherence: float = 0.5, chaos: float = 0.3):
        self.coherence = coherence
        self.chaos = chaos
        self.beta_res = self._compute_beta_res()
    
    def _compute_beta_res(self) -> float:
        """
        Compute optimal coupling β from Nash equilibrium.
        
        β_res = ((bγ - aδ) / (2c))^(1/3)
        where U = aC - bλ - cβ²
        """
        a, b, c = 1.0, 1.0, 1.0  # Utility weights
        gamma = self.coherence
        delta = 0.5  # Decay rate
        
        numerator = b * gamma - a * delta
        denominator = 2 * c
        
        if denominator > 0:
            beta_cubed = numerator / denominator
            beta_res = np.sign(beta_cubed) * (abs(beta_cubed) ** (1/3))
        else:
            beta_res = 0.0
        
        return beta_res
    
    def solve_nash(self, n_iterations: int = 50) -> dict:
        """Solve Nash equilibrium iteratively."""
        beta_history = [self.beta_res]
        utility_history = []
        
        for i in range(n_iterations):
            # Current utility
            utility = (1.0 * self.coherence - 
                      1.0 * self.chaos - 
                      1.0 * self.beta_res ** 2)
            utility_history.append(utility)
            
            # Update beta (gradient descent toward Nash)
            if i < n_iterations - 1:
                # Gradient: dU/dβ = -2cβ
                gradient = -2.0 * self.beta_res
                learning_rate = 0.01
                self.beta_res += learning_rate * gradient
                beta_history.append(self.beta_res)
        
        return {
            'beta_history': beta_history,
            'utility_history': utility_history,
            'final_beta': self.beta_res,
            'final_utility': utility_history[-1] if utility_history else 0.0
        }
    
    def visualize(self, output_file: str = ".out/nash_solver.png"):
        """Visualize Nash equilibrium convergence."""
        results = self.solve_nash()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Beta convergence
        ax1.plot(results['beta_history'], 'b-', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('β (Coupling)')
        ax1.set_title('Nash Equilibrium: β Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Utility evolution
        ax2.plot(results['utility_history'], 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Utility U = aC - bλ - cβ²')
        ax2.set_title('Utility Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")
        print(f"Final β: {results['final_beta']:.6f}")
        print(f"Final Utility: {results['final_utility']:.6f}")

def main():
    """Run interactive computational demonstration."""
    print("=" * 70)
    print("INTERACTIVE COMPUTATIONAL DEMONSTRATION")
    print("=" * 70)
    print()
    
    Path(".out").mkdir(exist_ok=True)
    
    # 1. Pascal's triangle
    print("1. Pascal's Triangle with Sin/Cos Path Splitting")
    print("-" * 70)
    pascal = PascalTriangleVisualizer(n_rows=10)
    pascal.visualize()
    print()
    
    # 2. Tan rotation
    print("2. tan(θ) Rotation on Unit Circle")
    print("-" * 70)
    tan_viz = TanRotationVisualizer()
    tan_viz.visualize()
    print()
    
    # 3. Attention mapping
    print("3. Transformer Attention → Angular Projections")
    print("-" * 70)
    attention = AttentionAngularMapper()
    attention.visualize()
    print()
    
    # 4. Nash solver
    print("4. Nash Equilibrium Solver (Guardian-Generator)")
    print("-" * 70)
    nash = NashEquilibriumSolver(coherence=0.5, chaos=0.3)
    nash.visualize()
    print()
    
    print("=" * 70)
    print("All visualizations saved to .out/")
    print("=" * 70)

if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        main()
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        print("Generating ASCII visualizations instead...")
        
        # ASCII fallback
        pascal = PascalTriangleVisualizer(n_rows=8)
        print("\nPascal's Triangle (first 8 rows):")
        for i in range(8):
            row_str = " ".join([str(pascal.triangle[i, j]) 
                              for j in range(i + 1)])
            print(f"{' ' * (8 - i)}{row_str}")







