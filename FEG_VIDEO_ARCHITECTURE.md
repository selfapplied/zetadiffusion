# FEG Video: Executable Video Architecture

**Status:** ✅ Prototype Complete

---

## Vision

**Executable video** where frames are computed via CSS/SVG declarations that evolve according to FEG topology parameters.

This isn't just a codec—it's **programmable video infrastructure**.

---

## Key Advantages

### 1. Extreme Compression
- **<1KB files** for complex visualizations
- Store topology transitions, not pixels
- Most frames computed from transformation rules

### 2. Infinite Resolution
- Render at any viewport size
- No pixelation at zoom
- Scalable vector graphics

### 3. Editable & Composable
- Video is a DOM (inspect element on frames)
- Tweak topology parameters
- Fork timelines
- Combine FEG videos like HTML

### 4. Version Controllable
- Diff topology transitions
- Git-friendly (text-based)
- Merge conflicts resolvable

### 5. Interactive
- Bind genus evolution to user input
- Real-time parameter adjustment
- Procedural generation from seeds

---

## Architecture

### Format: `.feg` XML Schema

```xml
<feg-video duration="120s" fps="30" resolution="1920x1080">
  <topology-state g="0" χ="2" H="0.7" timestamp="0s"/>
  
  <keyframe t="0s">
    <manifold-declaration>
      <style>
        .background { fill: radial-gradient(...); }
        .field-line { stroke: hsl(...); }
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

### Components

1. **Topology States**: Genus (g), Euler characteristic (χ), Hurst field (H)
2. **Keyframes**: CSS/SVG declarations at specific timestamps
3. **Transitions**: Style deltas for topology jumps (genus changes)
4. **Compiler**: Converts topology → CSS/SVG
5. **Player**: Browser-based renderer with CSS Houdini + Web Animations API

---

## Implementation Status

### ✅ Completed

- [x] XML schema definition (`feg_video/schema.py`)
- [x] Topology → CSS compiler (`feg_video/compiler.py`)
- [x] Browser player prototype (`feg_video/player.html`)
- [x] Demo: Heartbeat visualization (<1KB .feg file)

### 🚧 In Progress

- [ ] CSS Houdini Paint API integration
- [ ] Web Animations API for transitions
- [ ] Real-time genus visualizer
- [ ] Export from existing visualizations

### 📋 Planned

- [ ] ML upsampling for live-action reconstruction
- [ ] Batch conversion tools
- [ ] Editor interface
- [ ] Performance optimization

---

## Technical Feasibility

### Possible ✅

- **CSS Houdini Paint API**: Programmatic rendering
- **Web Animations API**: Smooth transitions
- **SVG**: Vector graphics support
- **Canvas API**: Fallback rendering

### Performance Considerations

- **Simple geometric animations**: Excellent (UI demos, data viz, abstract art)
- **Live-action video**: Computationally intensive, requires ML upsampling
- **Complex topologies**: May need GPU acceleration

---

## Strategic Value

### Market Position

**Not "better H.265"** → **"Programmable video infrastructure"**

### Use Cases

1. **UI Animations**: Compress design system animations
2. **Data Visualization**: Animated charts and graphs
3. **Abstract Art**: Procedural generative art
4. **Educational Content**: Mathematical visualizations
5. **Interactive Media**: Games, simulations

### Killer Demo

**Real-time genus visualizer** that exports playback-ready FEG files.

Show that existing FEG Heartbeat viz could be compressed to **<1KB .feg file**.

---

## Development Path

1. ✅ **Define .feg XML schema** (2 days) - DONE
2. ✅ **Build CSS player prototype** (3 days) - DONE
3. 🚧 **Implement topology → style compiler** (1 week) - IN PROGRESS
4. 📋 **Benchmark compression on geometric animations** (2 days)

---

## Files

- `feg_video/schema.py` - XML schema definition
- `feg_video/compiler.py` - Topology → CSS compiler
- `feg_video/player.html` - Browser player
- `feg_video/demo.py` - Demo generator
- `.out/heartbeat.feg` - Example output (<1KB)

---

## Next Steps

1. Enhance CSS compiler with more topology patterns
2. Add SVG path generation for complex curves
3. Integrate with existing FEG visualizations
4. Benchmark on real animation datasets
5. Build editor interface

---

**This repositions FEG from compression to executable media infrastructure.**




