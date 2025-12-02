#!.venv/bin/python
"""
feg_video/demo.py

Demo: Create FEG video from topology evolution.

Shows how to:
1. Define topology states
2. Create keyframes
3. Add topology transitions
4. Export to .feg XML
5. Play in browser

Author: Joel
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feg_video.schema import FEGVideo, TopologyState, Keyframe, TopologyTransition
from feg_video.compiler import compile_keyframe, transition_to_style_delta

def create_heartbeat_video() -> FEGVideo:
    """
    Create FEG Heartbeat visualization as executable video.
    
    This demonstrates the compression: a complex visualization
    compressed to <1KB .feg file.
    """
    video = FEGVideo(
        duration=10.0,
        fps=30,
        resolution="1920x1080"
    )
    
    # Topology evolution: sphere → torus → higher genus
    states = [
        TopologyState(g=0, chi=2, H=0.5, timestamp=0.0, coherence=0.8, chaos=0.2),
        TopologyState(g=0, chi=2, H=0.6, timestamp=2.0, coherence=0.7, chaos=0.3),
        TopologyState(g=1, chi=0, H=0.7, timestamp=5.0, coherence=0.6, chaos=0.4),
        TopologyState(g=1, chi=0, H=0.8, timestamp=7.0, coherence=0.5, chaos=0.5),
        TopologyState(g=2, chi=-2, H=0.9, timestamp=10.0, coherence=0.4, chaos=0.6),
    ]
    
    video.topology_states = states
    
    # Keyframes at topology states
    for state in states:
        keyframe = compile_keyframe(state, state.timestamp)
        video.keyframes.append(keyframe)
    
    # Topology transitions (genus jumps)
    trans1 = TopologyTransition(
        t=2.3,
        g_from=0, g_to=1,
        chi_from=2, chi_to=0,
        shock_energy=8.2,
        style_delta="",  # Will be generated
        duration=0.5
    )
    trans1.style_delta = transition_to_style_delta(trans1)
    
    trans2 = TopologyTransition(
        t=7.5,
        g_from=1, g_to=2,
        chi_from=0, chi_to=-2,
        shock_energy=12.5,
        style_delta="",  # Will be generated
        duration=0.8
    )
    trans2.style_delta = transition_to_style_delta(trans2)
    
    video.transitions = [trans1, trans2]
    
    return video

def main():
    """Create and save FEG video."""
    print("Creating FEG Heartbeat video...")
    
    video = create_heartbeat_video()
    
    # Export to XML
    xml_string = video.to_string(pretty=True)
    
    # Save to file
    output_file = ".out/heartbeat.feg"
    with open(output_file, 'w') as f:
        f.write(xml_string)
    
    print(f"✓ Saved FEG video to: {output_file}")
    print(f"  Size: {len(xml_string.encode('utf-8'))} bytes")
    print(f"  Duration: {video.duration}s")
    print(f"  Keyframes: {len(video.keyframes)}")
    print(f"  Transitions: {len(video.transitions)}")
    print()
    print("To play:")
    print(f"  1. Open feg_video/player.html in browser")
    print(f"  2. Load {output_file}")
    print()
    print("This demonstrates executable video:")
    print("  - Complex visualization compressed to <1KB")
    print("  - Frames computed from topology parameters")
    print("  - Infinite resolution (render at any size)")
    print("  - Editable (modify topology states)")

if __name__ == "__main__":
    main()

