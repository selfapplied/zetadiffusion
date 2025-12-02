# FEG Video: Executable Video Format

**Frames computed from topology parameters, not stored as pixels.**

---

## Quick Start

### 1. Open the Demo

```bash
# Open in browser
open feg_video/demo.html
# or
python3 -m http.server 8000
# then visit http://localhost:8000/feg_video/demo.html
```

### 2. Try Examples

The demo includes three example videos:
- **💓 Heartbeat**: Topology evolution (sphere → torus → higher genus)
- **🌀 Simple Wave**: Basic topology state
- **⚡ Shock Transition**: Genus jump with shock energy

Click any example to load and play it.

### 3. Load Your Own .feg Files

1. Click "📁 Load .feg File" button
2. Select a `.feg` file
3. Video loads and plays automatically

---

## Creating FEG Videos

### Python API

```python
from feg_video.schema import FEGVideo, TopologyState, Keyframe
from feg_video.compiler import compile_keyframe

# Create video
video = FEGVideo(duration=10.0, fps=30, resolution="1920x1080")

# Add topology state
state = TopologyState(g=0, chi=2, H=0.5, timestamp=0.0)
video.topology_states.append(state)

# Compile to keyframe
keyframe = compile_keyframe(state, 0.0)
video.keyframes.append(keyframe)

# Export to XML
xml_string = video.to_string()
with open('output.feg', 'w') as f:
    f.write(xml_string)
```

### Command Line

```bash
# Generate example video
python3 feg_video/demo.py

# Output: .out/heartbeat.feg
```

---

## File Format

FEG Video uses XML format:

```xml
<feg-video duration="10s" fps="30" resolution="1920x1080">
  <topology-state g="0" χ="2" H="0.5" timestamp="0s"/>
  
  <keyframe t="0s">
    <manifold-declaration>
      <style>
        .background { fill: radial-gradient(...); }
        .field-line { stroke: hsl(180, 70%, 60%); }
      </style>
    </manifold-declaration>
  </keyframe>
  
  <topology-transition t="2.3s" g="0→1" χ="2→0" shock-energy="8.2">
    <style-delta>
      .field-line { transform: rotateY(90deg); }
    </style-delta>
  </topology-transition>
</feg-video>
```

---

## Features

- **Extreme Compression**: <1KB for complex visualizations
- **Infinite Resolution**: Render at any size
- **Editable**: Modify topology parameters
- **Version Controllable**: Diff-friendly XML
- **Interactive**: Bind to user input

---

## Files

- `demo.html` - Interactive browser demo
- `player.html` - Basic player (for embedding)
- `schema.py` - XML schema definition
- `compiler.py` - Topology → CSS compiler
- `demo.py` - Python demo generator

---

## Next Steps

1. Open `demo.html` in browser
2. Try the example videos
3. Create your own with Python API
4. Integrate with existing visualizations




