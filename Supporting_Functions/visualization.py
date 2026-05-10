"""
Beautiful Neural Network Visualization Module
Provides:
1. Interactive network architecture visualization
2. Real-time training animation with forward/backward pass
3. Data flow animation
4. Loss and metrics tracking
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Tuple, Any, Optional
import time
from datetime import datetime


class NetworkVisualizer:
    """Creates beautiful interactive neural network visualizations"""
    
    def __init__(self, structure: List[Dict], input_size: int = None):
        """
        Initialize network visualizer
        
        Args:
            structure: List of layer dicts with 'type' and 'neurons' keys
            input_size: Number of input features
        """
        self.structure = structure
        self.input_size = input_size
        self.colors = {
            'input': '#00D9FF',      # Cyan
            'dense': '#FF6B9D',      # Pink
            'activation': '#FFA502', # Orange
            'output': '#2ECC71',     # Green
            'connection': '#34495E', # Dark gray
            'bg': '#ECF0F1',         # Light gray
        }
        
    def visualize_architecture(self, show_layer_info: bool = True) -> go.Figure:
        """
        Create beautiful static visualization of network architecture
        
        Args:
            show_layer_info: Whether to show neuron count and activation info
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Calculate positions
        layer_spacing = 150
        max_neurons = max(d.get('neurons', 3) for d in self.structure) if self.structure else 3
        neuron_spacing = 60 / (max_neurons / 3)
        
        positions = []
        node_traces = []
        edge_traces = []
        
        # Create layer positions
        all_layers = []
        
        # Add input layer
        if self.input_size:
            all_layers.append({'type': 'Input', 'neurons': self.input_size, 'index': -1})
        
        for idx, layer in enumerate(self.structure):
            layer['index'] = idx
            all_layers.append(layer)
        
        # Draw layers
        for layer_idx, layer in enumerate(all_layers):
            x = layer_idx * layer_spacing
            neurons_count = layer.get('neurons', 3)
            
            # Limit display of neurons for clarity
            display_neurons = min(neurons_count, 8)
            
            y_positions = np.linspace(
                -display_neurons * neuron_spacing / 2,
                display_neurons * neuron_spacing / 2,
                display_neurons
            )
            positions.append([(x, y) for y in y_positions])
            
            # Determine node color
            if layer['type'] == 'Input' or layer['type'] == 'input':
                color = self.colors['input']
                name = f"Input\n({neurons_count})"
            elif layer['type'] == 'Activation':
                color = self.colors['activation']
                name = f"{layer.get('activation', 'Activation')}"
            elif layer['type'] == 'Dense':
                color = self.colors['dense']
                name = f"Dense\n({neurons_count})"
            else:
                color = self.colors['output']
                name = layer.get('type', 'Layer')
            
            # Add nodes with glow effect
            x_coords = [x] * len(y_positions)
            y_coords = y_positions
            
            # Draw connection edges first (behind nodes)
            if layer_idx < len(all_layers) - 1:
                current_layer = positions[-1]
                next_layer = positions[layer_idx + 1] if layer_idx + 1 < len(positions) else []
                
                if next_layer:
                    for (x1, y1) in current_layer:
                        for (x2, y2) in next_layer:
                            edge_traces.append(
                                go.Scatter(
                                    x=[x1, x2], y=[y1, y2],
                                    mode='lines',
                                    line=dict(
                                        color=self.colors['connection'],
                                        width=1.5,
                                        dash='solid'
                                    ),
                                    hoverinfo='none',
                                    showlegend=False,
                                    opacity=0.4
                                )
                            )
            
            # Add neuron nodes
            node_traces.append(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=25,
                        color=color,
                        line=dict(
                            color='white',
                            width=2.5
                        ),
                        opacity=0.9,
                        symbol='circle'
                    ),
                    text=[name] * len(y_positions),
                    textposition='top center' if layer_idx % 2 == 0 else 'bottom center',
                    textfont=dict(
                        size=9,
                        color='white',
                        family='Arial Black'
                    ),
                    name=f"Layer {layer_idx}",
                    hovertext=[f"Neurons: {neurons_count}<br>Type: {layer['type']}" for _ in y_positions],
                    hoverinfo='text',
                    showlegend=True,
                    customdata=[[layer['type'], neurons_count]] * len(y_positions)
                )
            )
        
        # Add edges first (background)
        for edge_trace in edge_traces:
            fig.add_trace(edge_trace)
        
        # Add nodes on top
        for node_trace in node_traces:
            fig.add_trace(node_trace)
        
        # Update layout with professional styling
        fig.update_layout(
            title=dict(
                text="<b>Neural Network Architecture</b>",
                font=dict(size=24, color='#2C3E50'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=1200,
            font=dict(family='Arial', size=11),
            hoverlabel=dict(
                bgcolor='white',
                font_size=13,
                font_family='Arial'
            )
        )
        
        return fig
    
    def create_training_dashboard(self) -> Tuple[go.Figure, Dict]:
        """
        Create dashboard for real-time training metrics
        
        Returns:
            Tuple of (figure, metrics_dict)
        """
        metrics = {
            'epochs': [],
            'loss': [],
            'accuracy': [],
            'forward_time': [],
            'backward_time': [],
            'learning_rate': 0.01
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Loss Curve</b>',
                '<b>Accuracy Progression</b>',
                '<b>Forward & Backward Pass Time</b>',
                '<b>Training Statistics</b>'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        return fig, metrics


class TrainingAnimator:
    """Handles real-time training visualization with animations"""
    
    def __init__(self, container, network_structure: List[Dict], total_epochs: int):
        """
        Initialize training animator
        
        Args:
            container: Streamlit container to display animations
            network_structure: Neural network layer structure
            total_epochs: Total number of epochs to train
        """
        self.container = container
        self.structure = network_structure
        self.total_epochs = total_epochs
        self.epoch_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'forward_time': [],
            'backward_time': []
        }
        self.visualizer = NetworkVisualizer(network_structure)
        
    def create_epoch_progress_display(self) -> Tuple[Any, Any, Any, Any]:
        """Create placeholders for real-time metric updates"""
        col1, col2, col3, col4 = self.container.columns(4)
        
        with col1:
            epoch_placeholder = st.empty()
        with col2:
            loss_placeholder = st.empty()
        with col3:
            accuracy_placeholder = st.empty()
        with col4:
            time_placeholder = st.empty()
        
        return epoch_placeholder, loss_placeholder, accuracy_placeholder, time_placeholder
    
    def update_epoch_display(self, epoch_ph, loss_ph, acc_ph, time_ph, 
                            epoch: int, loss: float, accuracy: float, elapsed_time: float):
        """Update epoch display with current metrics"""
        with epoch_ph.container():
            st.metric(
                "📊 Epoch",
                f"{epoch}/{self.total_epochs}",
                delta=f"{(epoch/self.total_epochs*100):.1f}%"
            )
        
        with loss_ph.container():
            st.metric("📉 Loss", f"{loss:.6f}")
        
        with acc_ph.container():
            st.metric("📈 Accuracy", f"{accuracy:.4f}")
        
        with time_ph.container():
            st.metric("⏱️ Time", f"{elapsed_time:.2f}s")
    
    def animate_forward_pass(self, batch_data: np.ndarray, layer_count: int) -> None:
        """
        Animate data flowing through forward pass
        
        Args:
            batch_data: Input batch data
            layer_count: Number of layers in network
        """
        fig = go.Figure()
        
        # Create frames for animation
        frames = []
        
        for frame_idx in range(layer_count + 1):
            # Calculate which connections to highlight
            x_positions = []
            y_positions = []
            colors = []
            
            for layer_idx in range(frame_idx + 1):
                layer_spacing = 150
                x = layer_idx * layer_spacing
                
                # Create gradient from input to output
                intensity = (layer_idx / (layer_count)) * 255
                color = f'rgb({int(intensity)}, {int(128)}, {int(255-intensity)})'
                
                # Add neurons for this layer
                for neuron_idx in range(3):
                    y = neuron_idx * 60 - 60
                    x_positions.append(x)
                    y_positions.append(y)
                    colors.append(color)
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=x_positions,
                        y=y_positions,
                        mode='markers',
                        marker=dict(
                            size=20,
                            color=colors,
                            line=dict(color='white', width=2)
                        )
                    )
                ],
                name=f"Step {frame_idx}"
            )
            frames.append(frame)
        
        fig.frames = frames
        
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers',
                marker=dict(size=20, color=colors, line=dict(color='white', width=2))
            )
        )
        
        # Create play button animation
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ Forward Pass',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 300, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    }
                ]
            }],
            title='Forward Pass Animation',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_real_time_metrics_plot(self, epochs_list: List[int], 
                                     loss_list: List[float],
                                     accuracy_list: List[float]) -> go.Figure:
        """
        Create real-time updating metrics plot
        
        Args:
            epochs_list: List of epoch numbers
            loss_list: List of loss values
            accuracy_list: List of accuracy values
            
        Returns:
            Plotly figure with dual-axis plot
        """
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add loss curve
        fig.add_trace(
            go.Scatter(
                x=epochs_list,
                y=loss_list,
                name='Loss',
                mode='lines+markers',
                line=dict(
                    color='#FF6B9D',
                    width=3,
                    shape='spline'
                ),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 157, 0.2)',
                hovertemplate='<b>Epoch %{x}</b><br>Loss: %{y:.6f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add accuracy curve
        fig.add_trace(
            go.Scatter(
                x=epochs_list,
                y=accuracy_list,
                name='Accuracy',
                mode='lines+markers',
                line=dict(
                    color='#2ECC71',
                    width=3,
                    shape='spline'
                ),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)',
                hovertemplate='<b>Epoch %{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(
            title_text='<b>Epochs</b>',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False
        )
        
        fig.update_yaxes(
            title_text='<b>Loss</b>',
            secondary_y=False,
            titlefont=dict(color='#FF6B9D'),
            tickfont=dict(color='#FF6B9D')
        )
        
        fig.update_yaxes(
            title_text='<b>Accuracy</b>',
            secondary_y=True,
            titlefont=dict(color='#2ECC71'),
            tickfont=dict(color='#2ECC71')
        )
        
        fig.update_layout(
            title={
                'text': '<b>Training Progress - Real Time Metrics</b>',
                'font': {'size': 20, 'color': '#2C3E50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='x unified',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial', size=12),
            margin=dict(t=80, b=60, l=80, r=80)
        )
        
        return fig
    
    def create_data_flow_animation(self, sample_data: np.ndarray) -> go.Figure:
        """
        Create animation showing data flowing through network
        
        Args:
            sample_data: Sample input data
            
        Returns:
            Animated Plotly figure
        """
        fig = go.Figure()
        
        # Create frames showing data propagating through layers
        frames = []
        layer_spacing = 150
        
        num_layers = len(self.structure) + 1
        num_samples = min(5, sample_data.shape[0])  # Show up to 5 samples
        
        for layer_idx in range(num_layers):
            frame_data = []
            
            # Draw all layers
            for l_idx in range(num_layers):
                x = l_idx * layer_spacing
                
                # Determine color intensity based on data flow
                if l_idx <= layer_idx:
                    intensity = (l_idx / num_layers) * 255
                    color = f'rgb(0, {int(150 + intensity/2)}, 255)'
                    size = 25
                    opacity = 0.95
                else:
                    color = 'rgb(200, 200, 200)'
                    size = 20
                    opacity = 0.4
                
                # Add neurons
                for neuron_idx in range(3):
                    y = neuron_idx * 60 - 60
                    frame_data.append(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode='markers',
                            marker=dict(
                                size=size,
                                color=color,
                                line=dict(color='white', width=2),
                                opacity=opacity
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
                
                # Add connections
                if l_idx < num_layers - 1:
                    for neuron_idx in range(3):
                        y1 = neuron_idx * 60 - 60
                        for next_neuron in range(3):
                            y2 = next_neuron * 60 - 60
                            opacity_conn = 0.5 if l_idx <= layer_idx else 0.1
                            frame_data.append(
                                go.Scatter(
                                    x=[x, (l_idx + 1) * layer_spacing],
                                    y=[y1, y2],
                                    mode='lines',
                                    line=dict(
                                        color='rgb(100, 150, 200)',
                                        width=1
                                    ),
                                    opacity=opacity_conn,
                                    showlegend=False,
                                    hoverinfo='skip'
                                )
                            )
            
            frames.append(go.Frame(data=frame_data, name=f"Layer {layer_idx}"))
        
        fig.frames = frames
        
        # Add initial trace
        initial_data = []
        for l_idx in range(num_layers):
            x = l_idx * layer_spacing
            for neuron_idx in range(3):
                y = neuron_idx * 60 - 60
                initial_data.append(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='rgb(200, 200, 200)',
                            line=dict(color='white', width=2),
                            opacity=0.4
                        ),
                        showlegend=False
                    )
                )
        
        fig.add_traces(initial_data)
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'buttons': [
                    {
                        'label': '▶ Data Flow Forward',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                        }]
                    },
                    {
                        'label': '⏹ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            title={
                'text': '<b>Data Flow Animation Through Network</b>',
                'font': {'size': 18, 'color': '#2C3E50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            margin=dict(t=80, b=40, l=40, r=40),
            showlegend=False,
            hovermode='closest'
        )
        
        return fig


def visualize_network_architecture(structure: List[Dict], input_size: int = None) -> go.Figure:
    """
    Quick function to visualize network architecture
    
    Args:
        structure: Network layer structure
        input_size: Number of input features
        
    Returns:
        Plotly figure
    """
    visualizer = NetworkVisualizer(structure, input_size)
    return visualizer.visualize_architecture()


def create_training_metrics_display(epochs: List[int], 
                                   losses: List[float],
                                   accuracies: List[float]) -> go.Figure:
    """
    Create training metrics visualization
    
    Args:
        epochs: List of epoch numbers
        losses: List of loss values
        accuracies: List of accuracy values
        
    Returns:
        Plotly figure with metrics
    """
    animator = TrainingAnimator(None, [], len(epochs))
    return animator.create_real_time_metrics_plot(epochs, losses, accuracies)


def create_training_animation_container(structure: List[Dict], total_epochs: int) -> TrainingAnimator:
    """
    Create and return a training animator for real-time updates
    
    Args:
        structure: Network structure
        total_epochs: Total epochs to train
        
    Returns:
        TrainingAnimator instance
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Training Progress")
        progress_container = st.container()
    
    with col2:
        st.markdown("### 📈 Metrics")
        metrics_container = st.container()
    
    animator = TrainingAnimator(progress_container, structure, total_epochs)
    return animator, metrics_container
