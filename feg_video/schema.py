"""
feg_video/schema.py

FEG Video XML Schema Definition

Executable video format where frames are computed via CSS/SVG declarations
that evolve according to FEG topology parameters.

Author: Joel
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from xml.etree.ElementTree import Element, SubElement, tostring, fromstring
import math

@dataclass
class String:
    """Fundamental oscillator (standing wave)."""
    i: int  # String index
    omega: float  # Frequency (ω)
    phi: float  # Phase (φ)
    amp: float  # Amplitude (A)
    mode: str = "fundamental"  # Mode type
    
    def evaluate(self, t: float) -> float:
        """Evaluate string(i, t) = A · sin(ω·t + φ)."""
        return self.amp * math.sin(self.omega * t + self.phi)

@dataclass
class TopologyState:
    """Topology state at a given timestamp."""
    g: int  # Genus
    chi: int  # Euler characteristic (2 - 2g)
    H: float  # Hurst field (memory)
    timestamp: float  # Time in seconds
    coherence: Optional[float] = None
    chaos: Optional[float] = None
    stress: Optional[float] = None

@dataclass
class Keyframe:
    """Keyframe with CSS/SVG declarations."""
    t: float  # Timestamp
    css: str  # CSS declarations for this frame
    svg: Optional[str] = None  # SVG markup (if needed)
    topology: Optional[TopologyState] = None

@dataclass
class TopologyTransition:
    """Topology transition (genus jump, shock event)."""
    t: float  # Timestamp
    g_from: int  # Starting genus
    g_to: int  # Ending genus
    chi_from: int  # Starting Euler characteristic
    chi_to: int  # Ending Euler characteristic
    shock_energy: float  # Energy released during transition
    style_delta: str  # CSS style changes for transition
    duration: Optional[float] = None  # Transition duration (default: auto)

@dataclass
class FEGVideo:
    """FEG Video document."""
    duration: float  # Total duration in seconds
    fps: int  # Frames per second
    resolution: str  # e.g., "1920x1080"
    topology_states: List[TopologyState] = field(default_factory=list)
    keyframes: List[Keyframe] = field(default_factory=list)
    transitions: List[TopologyTransition] = field(default_factory=list)
    strings: List[String] = field(default_factory=list)  # Harmonic string basis
    
    def to_xml(self) -> Element:
        """Convert to XML Element."""
        root = Element('feg-video')
        root.set('duration', f'{self.duration}s')
        root.set('fps', str(self.fps))
        root.set('resolution', self.resolution)
        
        # Strings (harmonic basis)
        if self.strings:
            strings_elem = SubElement(root, 'feg:strings')
            for s in self.strings:
                string_elem = SubElement(strings_elem, 'string')
                string_elem.set('i', str(s.i))
                string_elem.set('ω', str(s.omega))
                string_elem.set('φ', str(s.phi))
                string_elem.set('A', str(s.amp))
                if s.mode != "fundamental":
                    string_elem.set('mode', s.mode)
        
        # Topology states
        for state in self.topology_states:
            state_elem = SubElement(root, 'topology-state')
            state_elem.set('g', str(state.g))
            state_elem.set('χ', str(state.chi))
            state_elem.set('H', str(state.H))
            state_elem.set('timestamp', f'{state.timestamp}s')
            if state.coherence is not None:
                state_elem.set('coherence', str(state.coherence))
            if state.chaos is not None:
                state_elem.set('chaos', str(state.chaos))
            if state.stress is not None:
                state_elem.set('stress', str(state.stress))
        
        # Keyframes
        for keyframe in self.keyframes:
            kf_elem = SubElement(root, 'keyframe')
            kf_elem.set('t', f'{keyframe.t}s')
            
            if keyframe.topology:
                topo_elem = SubElement(kf_elem, 'topology-state')
                topo_elem.set('g', str(keyframe.topology.g))
                topo_elem.set('χ', str(keyframe.topology.chi))
                topo_elem.set('H', str(keyframe.topology.H))
            
            manifold = SubElement(kf_elem, 'manifold-declaration')
            style = SubElement(manifold, 'style')
            style.text = keyframe.css
            
            if keyframe.svg:
                svg_elem = SubElement(manifold, 'svg')
                svg_elem.text = keyframe.svg
        
        # Topology transitions
        for trans in self.transitions:
            trans_elem = SubElement(root, 'topology-transition')
            trans_elem.set('t', f'{trans.t}s')
            trans_elem.set('g', f'{trans.g_from}→{trans.g_to}')
            trans_elem.set('χ', f'{trans.chi_from}→{trans.chi_to}')
            trans_elem.set('shock-energy', str(trans.shock_energy))
            if trans.duration:
                trans_elem.set('duration', f'{trans.duration}s')
            
            style_delta = SubElement(trans_elem, 'style-delta')
            style_delta.text = trans.style_delta
        
        return root
    
    def to_string(self, pretty: bool = True) -> str:
        """Convert to XML string."""
        root = self.to_xml()
        if pretty:
            # Basic pretty printing
            from xml.dom import minidom
            rough_string = tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")
        else:
            return tostring(root, 'unicode')
    
    @classmethod
    def from_xml(cls, xml_string: str) -> 'FEGVideo':
        """Parse from XML string."""
        root = fromstring(xml_string)
        
        duration = float(root.get('duration', '0').rstrip('s'))
        fps = int(root.get('fps', '30'))
        resolution = root.get('resolution', '1920x1080')
        
        video = cls(duration=duration, fps=fps, resolution=resolution)
        
        # Parse strings
        strings_elem = root.find('feg:strings')
        if strings_elem is None:
            # Try without namespace
            strings_elem = root.find('strings')
        
        if strings_elem is not None:
            for string_elem in strings_elem.findall('string'):
                phi_str = string_elem.get('φ', '0.0')
                # Handle π notation
                if 'π' in phi_str:
                    phi_val = float(phi_str.replace('π', '').strip() or '1') * math.pi
                else:
                    phi_val = float(phi_str)
                
                s = String(
                    i=int(string_elem.get('i', '0')),
                    omega=float(string_elem.get('ω', '1.0')),
                    phi=phi_val,
                    amp=float(string_elem.get('A', '1.0')),
                    mode=string_elem.get('mode', 'fundamental')
                )
                video.strings.append(s)
        
        # Parse topology states
        for state_elem in root.findall('topology-state'):
            state = TopologyState(
                g=int(state_elem.get('g', '0')),
                chi=int(state_elem.get('χ', '2')),
                H=float(state_elem.get('H', '0.5')),
                timestamp=float(state_elem.get('timestamp', '0').rstrip('s')),
                coherence=float(state_elem.get('coherence')) if state_elem.get('coherence') else None,
                chaos=float(state_elem.get('chaos')) if state_elem.get('chaos') else None,
                stress=float(state_elem.get('stress')) if state_elem.get('stress') else None
            )
            video.topology_states.append(state)
        
        # Parse keyframes
        for kf_elem in root.findall('keyframe'):
            t = float(kf_elem.get('t', '0').rstrip('s'))
            
            manifold = kf_elem.find('manifold-declaration')
            css = ""
            svg = None
            
            if manifold is not None:
                style = manifold.find('style')
                if style is not None and style.text:
                    css = style.text
                
                svg_elem = manifold.find('svg')
                if svg_elem is not None and svg_elem.text:
                    svg = svg_elem.text
            
            # Parse topology from keyframe if present
            topo_elem = kf_elem.find('topology-state')
            topology = None
            if topo_elem is not None:
                topology = TopologyState(
                    g=int(topo_elem.get('g', '0')),
                    chi=int(topo_elem.get('χ', '2')),
                    H=float(topo_elem.get('H', '0.5')),
                    timestamp=t
                )
            
            keyframe = Keyframe(t=t, css=css, svg=svg, topology=topology)
            video.keyframes.append(keyframe)
        
        # Parse transitions
        for trans_elem in root.findall('topology-transition'):
            t = float(trans_elem.get('t', '0').rstrip('s'))
            g_str = trans_elem.get('g', '0→0')
            g_from, g_to = map(int, g_str.split('→'))
            chi_str = trans_elem.get('χ', '2→2')
            chi_from, chi_to = map(int, chi_str.split('→'))
            shock_energy = float(trans_elem.get('shock-energy', '0'))
            duration = None
            if trans_elem.get('duration'):
                duration = float(trans_elem.get('duration').rstrip('s'))
            
            style_delta_elem = trans_elem.find('style-delta')
            style_delta = style_delta_elem.text if style_delta_elem is not None else ""
            
            trans = TopologyTransition(
                t=t, g_from=g_from, g_to=g_to,
                chi_from=chi_from, chi_to=chi_to,
                shock_energy=shock_energy, style_delta=style_delta,
                duration=duration
            )
            video.transitions.append(trans)
        
        return video


