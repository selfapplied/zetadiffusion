"""
feg_video: Executable Video Format

FEG Video is executable video where frames are computed via CSS/SVG declarations
that evolve according to FEG topology parameters.

Author: Joel
"""

from feg_video.schema import FEGVideo, TopologyState, Keyframe, TopologyTransition
from feg_video.compiler import topology_to_css, transition_to_style_delta, compile_keyframe

__all__ = [
    'FEGVideo',
    'TopologyState',
    'Keyframe',
    'TopologyTransition',
    'topology_to_css',
    'transition_to_style_delta',
    'compile_keyframe',
]




