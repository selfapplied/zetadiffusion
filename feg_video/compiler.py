"""
feg_video/compiler.py

Topology → Style Compiler

Converts FEG topology parameters to CSS/SVG declarations.

Author: Joel
"""

from typing import Dict, Optional
import math
from feg_video.schema import TopologyState, Keyframe, TopologyTransition

def topology_to_css(state: TopologyState, width: int = 1920, height: int = 1080) -> str:
    """
    Convert topology state to CSS declarations.
    
    Args:
        state: TopologyState with genus, Euler characteristic, Hurst field
        width: Canvas width
        height: Canvas height
    
    Returns:
        CSS string with declarations
    """
    g = state.g
    chi = state.chi
    H = state.H
    
    # Base background gradient based on genus
    # Sphere (g=0): radial gradient
    # Torus (g=1): linear gradient with rotation
    # Higher genus: more complex patterns
    
    if g == 0:
        # Sphere: radial gradient
        bg_css = f"""
.background {{
  fill: radial-gradient(circle at {width//2}px {height//2}px, 
    hsl({int(H * 360)}, 70%, 20%), 
    hsl({int(H * 360 + 60)}, 70%, 10%));
  width: {width}px;
  height: {height}px;
}}
"""
    elif g == 1:
        # Torus: linear gradient with rotation
        angle = H * 360
        bg_css = f"""
.background {{
  fill: linear-gradient({angle}deg,
    hsl({int(H * 360)}, 70%, 30%),
    hsl({int(H * 360 + 120)}, 70%, 15%));
  width: {width}px;
  height: {height}px;
  transform: rotate({angle * 0.1}deg);
}}
"""
    else:
        # Higher genus: complex pattern
        bg_css = f"""
.background {{
  background: 
    repeating-linear-gradient(
      {H * 45}deg,
      hsl({int(H * 360)}, 70%, 25%) 0px,
      hsl({int(H * 360 + 60)}, 70%, 20%) {g * 10}px,
      hsl({int(H * 360 + 120)}, 70%, 15%) {g * 20}px
    );
  width: {width}px;
  height: {height}px;
}}
"""
    
    # Field lines based on Euler characteristic
    # chi = 2 - 2g determines connectivity
    num_lines = max(1, abs(chi))
    line_css = ""
    
    for i in range(num_lines):
        phase = (i / num_lines) * 2 * math.pi
        x1 = width // 2 + int(math.cos(phase) * width * 0.3)
        y1 = height // 2 + int(math.sin(phase) * height * 0.3)
        x2 = width // 2 + int(math.cos(phase + math.pi) * width * 0.3)
        y2 = height // 2 + int(math.sin(phase + math.pi) * height * 0.3)
        
        hue = int((H * 360 + phase * 180 / math.pi) % 360)
        
        line_css += f"""
.field-line-{i} {{
  stroke: hsl({hue}, 70%, 60%);
  stroke-width: {2 + H * 3}px;
  opacity: {0.6 + H * 0.4};
  d: path("M {x1},{y1} Q {width//2},{height//2} {x2},{y2}");
}}
"""
    
    # Hurst field visualization (memory texture)
    hurst_css = f"""
.hurst-field {{
  opacity: {H};
  filter: blur({(1 - H) * 10}px);
  mix-blend-mode: multiply;
}}
"""
    
    return bg_css + line_css + hurst_css

def transition_to_style_delta(trans: TopologyTransition, 
                             width: int = 1920, height: int = 1080) -> str:
    """
    Generate CSS style delta for topology transition.
    
    Args:
        trans: TopologyTransition with genus jump
        width: Canvas width
        height: Canvas height
    
    Returns:
        CSS string with transition styles
    """
    g_from = trans.g_from
    g_to = trans.g_to
    chi_from = trans.chi_from
    chi_to = trans.chi_to
    energy = trans.shock_energy
    
    # Shock energy affects transition intensity
    intensity = min(1.0, energy / 10.0)
    
    # Genus jump: sphere → torus
    if g_from == 0 and g_to == 1:
        return f"""
.field-line {{
  d: path("M {width//4},{height//2} A {width//4},{height//4} 0 1,1 {3*width//4},{height//2}");
  transform: rotateY({intensity * 180}deg);
  transition: transform {trans.duration or 0.5}s ease-in-out;
}}

.background {{
  transform: rotate({intensity * 360}deg);
  transition: transform {trans.duration or 0.5}s ease-in-out;
}}
"""
    
    # Torus → higher genus
    elif g_from == 1 and g_to > 1:
        return f"""
.field-line {{
  d: path("M {width//2},{height//2} m -{width//4},0 a {width//4},{height//4} 0 1,0 {width//2},0 a {width//4},{height//4} 0 1,0 -{width//2},0");
  transform: scale({1 + intensity * 0.5});
  transition: transform {trans.duration or 0.5}s ease-in-out;
}}
"""
    
    # Generic transition
    else:
        return f"""
.field-line {{
  transform: scale({1 + intensity * 0.3}) rotate({intensity * 180}deg);
  opacity: {0.7 + intensity * 0.3};
  transition: all {trans.duration or 0.5}s ease-in-out;
}}
"""

def compile_keyframe(state: TopologyState, t: float, 
                    width: int = 1920, height: int = 1080) -> Keyframe:
    """
    Compile topology state to keyframe.
    
    Args:
        state: TopologyState
        t: Timestamp
        width: Canvas width
        height: Canvas height
    
    Returns:
        Keyframe with CSS declarations
    """
    css = topology_to_css(state, width, height)
    
    return Keyframe(
        t=t,
        css=css,
        topology=state
    )




