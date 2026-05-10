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
        """Create a modern dark-themed neural network architecture visualization."""
        fig = go.Figure()

        # ── Build enriched layer list with parameter counts ──────────────────
        all_layers = []
        prev_n = self.input_size or 1
        total_params = 0

        if self.input_size:
            all_layers.append({
                'type': 'Input', 'neurons': self.input_size,
                'display_n': min(self.input_size, 7),
                'params': 0, 'activation': None,
            })

        for layer in self.structure:
            if layer['type'] == 'Dense':
                n = layer['neurons']
                p = prev_n * n + n
                total_params += p
                all_layers.append({
                    'type': 'Dense', 'neurons': n,
                    'display_n': min(n, 7),
                    'params': p, 'activation': None,
                })
                prev_n = n
            elif layer['type'] == 'Activation':
                all_layers.append({
                    'type': 'Activation', 'neurons': prev_n,
                    'display_n': min(prev_n, 7),
                    'params': 0, 'activation': layer.get('activation', 'ReLU'),
                })

        if not all_layers:
            fig.update_layout(
                paper_bgcolor='#0D1117', plot_bgcolor='#0D1117', height=300,
                annotations=[dict(
                    text='<b>Add layers to see the architecture</b>',
                    showarrow=False, font=dict(color='rgba(200,200,200,0.6)', size=16),
                    x=0.5, y=0.5, xref='paper', yref='paper'
                )]
            )
            return fig

        # ── Layout constants ─────────────────────────────────────────────────
        NEURON_GAP = 52
        LAYER_SEP = 230
        NODE_R = 18
        CARD_W = 134
        CARD_PAD = 38

        # Color palette: (node, glow, card_bg, card_border, conn)
        C = {
            'Input':      ('#00D9FF', 'rgba(0,217,255,0.28)',   'rgba(0,217,255,0.07)',   'rgba(0,217,255,0.55)',   'rgba(0,217,255,0.15)'),
            'Dense':      ('#D97BFF', 'rgba(217,123,255,0.28)', 'rgba(217,123,255,0.07)', 'rgba(217,123,255,0.55)', 'rgba(217,123,255,0.15)'),
            'Activation': ('#FFB830', 'rgba(255,184,48,0.28)',  'rgba(255,184,48,0.07)',  'rgba(255,184,48,0.55)',  'rgba(255,184,48,0.15)'),
        }

        # ── Compute y positions per layer ────────────────────────────────────
        layer_x = [i * LAYER_SEP for i in range(len(all_layers))]
        layer_y = {}
        for i, layer in enumerate(all_layers):
            dn = layer['display_n']
            layer_y[i] = [(j - (dn - 1) / 2.0) * NEURON_GAP for j in range(dn)]

        max_y_val = max(max(ys) for ys in layer_y.values())
        min_y_val = min(min(ys) for ys in layer_y.values())
        x_left  = layer_x[0]  - CARD_W / 2 - 20
        x_right = layer_x[-1] + CARD_W / 2 + 20

        # ── Subtle dot grid (background decoration) ──────────────────────────
        grid_xs, grid_ys = [], []
        step = 45
        gy_start = int(min_y_val - CARD_PAD - 140)
        gy_end   = int(max_y_val + CARD_PAD + 70)
        gx_start = int(x_left  - 10)
        gx_end   = int(x_right + 10)
        for gx in range(gx_start, gx_end, step):
            for gy in range(gy_start, gy_end, step):
                grid_xs.append(gx)
                grid_ys.append(gy)
        fig.add_trace(go.Scatter(
            x=grid_xs, y=grid_ys, mode='markers',
            marker=dict(size=1.8, color='rgba(255,255,255,0.055)', symbol='circle'),
            showlegend=False, hoverinfo='none',
        ))

        # ── Connections (S-curve via spline with 4 control points) ───────────
        for i in range(len(all_layers) - 1):
            sx, dx = layer_x[i], layer_x[i + 1]
            sy_list, dy_list = layer_y[i], layer_y[i + 1]
            conn_color = C.get(all_layers[i]['type'], C['Dense'])[4]

            pairs = [(sy, dy) for sy in sy_list for dy in dy_list]
            if len(pairs) > 30:
                step_c = max(1, len(pairs) // 22)
                pairs = pairs[::step_c]

            for sy, dy in pairs:
                t = 0.32  # bezier tension: how far inward the control points are
                fig.add_trace(go.Scatter(
                    x=[sx, sx + (dx - sx) * t, sx + (dx - sx) * (1 - t), dx],
                    y=[sy, sy,                  dy,                        dy],
                    mode='lines',
                    line=dict(color=conn_color, width=1.1, shape='spline'),
                    hoverinfo='none', showlegend=False,
                ))

        # ── Layer glass cards ─────────────────────────────────────────────────
        for i, layer in enumerate(all_layers):
            _, _, card_bg, card_border, _ = C.get(layer['type'], C['Dense'])
            ys = layer_y[i]
            top = max(ys) + CARD_PAD
            bot = min(ys) - CARD_PAD
            cx  = layer_x[i]
            # Outer card
            fig.add_shape(type='rect',
                x0=cx - CARD_W / 2, y0=bot, x1=cx + CARD_W / 2, y1=top,
                fillcolor=card_bg, line=dict(color=card_border, width=1.5),
                layer='below',
            )
            # Inner shimmer strip (top highlight)
            fig.add_shape(type='rect',
                x0=cx - CARD_W / 2 + 2, y0=top - 6, x1=cx + CARD_W / 2 - 2, y1=top - 2,
                fillcolor='rgba(255,255,255,0.06)', line=dict(width=0),
                layer='below',
            )

        # ── Neuron glow halos ─────────────────────────────────────────────────
        for i, layer in enumerate(all_layers):
            glow_c = C.get(layer['type'], C['Dense'])[1]
            ys = layer_y[i]
            fig.add_trace(go.Scatter(
                x=[layer_x[i]] * len(ys), y=ys,
                mode='markers',
                marker=dict(size=NODE_R * 2.7, color=glow_c, line=dict(width=0)),
                showlegend=False, hoverinfo='none',
            ))

        # ── Neuron bodies ─────────────────────────────────────────────────────
        for i, layer in enumerate(all_layers):
            node_c = C.get(layer['type'], C['Dense'])[0]
            ys = layer_y[i]
            n_shown  = layer['display_n']
            n_actual = layer['neurons']

            hover = []
            for j in range(n_shown):
                if layer['type'] == 'Input':
                    hover.append(f"<b>Input Feature {j + 1}</b><br>{n_actual} total features<br>0 parameters")
                elif layer['type'] == 'Activation':
                    hover.append(f"<b>{layer['activation']} Unit {j + 1}</b><br>Activation function<br>0 parameters")
                else:
                    hover.append(f"<b>Dense Neuron {j + 1}/{n_actual}</b><br>Trainable params: {layer['params']:,}<br>Weight matrix: {prev_n}×{n_actual}")

            fig.add_trace(go.Scatter(
                x=[layer_x[i]] * n_shown, y=ys,
                mode='markers+text',
                marker=dict(
                    size=NODE_R * 2,
                    color=node_c,
                    line=dict(color='rgba(255,255,255,0.45)', width=2),
                ),
                text=[str(j + 1) for j in range(n_shown)],
                textfont=dict(size=9, color='white', family='Arial Black'),
                textposition='middle center',
                hovertext=hover, hoverinfo='text',
                showlegend=False,
            ))

            if n_actual > n_shown:
                fig.add_annotation(
                    x=layer_x[i], y=max(ys) + NEURON_GAP * 0.62,
                    text=f"+{n_actual - n_shown} more",
                    showarrow=False,
                    font=dict(size=9, color='rgba(200,200,200,0.5)', family='Arial'),
                )

        # ── Info panels below each card ───────────────────────────────────────
        for i, layer in enumerate(all_layers):
            node_c, _, _, card_border, _ = C.get(layer['type'], C['Dense'])
            ys    = layer_y[i]
            bot   = min(ys) - CARD_PAD
            cx    = layer_x[i]

            if layer['type'] == 'Input':
                badge = 'INPUT'
                detail = f"{layer['neurons']} features"
                sub    = '─── no params ───'
            elif layer['type'] == 'Dense':
                badge = 'DENSE'
                detail = f"{layer['neurons']} neurons"
                sub    = f"{layer['params']:,} parameters"
            else:
                badge = layer['activation'].upper()
                detail = 'activation fn'
                sub    = '─── no params ───'

            fig.add_annotation(
                x=cx, y=bot - 10,
                text=(f"<b style='color:{node_c};font-size:11px;letter-spacing:1px'>{badge}</b><br>"
                      f"<span style='color:#ccc;font-size:10px'>{detail}</span><br>"
                      f"<span style='color:#888;font-size:9px'>{sub}</span>"),
                showarrow=False, align='center',
                font=dict(size=11, color=node_c, family='Courier New'),
                bgcolor='rgba(13,17,23,0.85)',
                bordercolor=card_border, borderwidth=1, borderpad=9,
                yanchor='top',
            )

        # ── Layer index chips (above each card) ───────────────────────────────
        for i, layer in enumerate(all_layers):
            node_c = C.get(layer['type'], C['Dense'])[0]
            ys  = layer_y[i]
            top = max(ys) + CARD_PAD
            cx  = layer_x[i]
            fig.add_annotation(
                x=cx, y=top + 10,
                text=f"<b>Layer {i + 1}</b>",
                showarrow=False,
                font=dict(size=10, color='rgba(210,210,210,0.55)', family='Arial'),
                yanchor='bottom',
            )

        # ── Legend chips (top-right corner) ──────────────────────────────────
        legend_items = [
            ('● Input',      C['Input'][0]),
            ('● Dense',      C['Dense'][0]),
            ('● Activation', C['Activation'][0]),
        ]
        legend_x = x_right - 10
        legend_y_start = max_y_val + CARD_PAD + 8
        for li, (label, color) in enumerate(legend_items):
            fig.add_annotation(
                x=legend_x, y=legend_y_start - li * 22,
                text=f"<span style='color:{color}'>{label}</span>",
                showarrow=False, align='right',
                font=dict(size=11, family='Arial', color=color),
                xanchor='right', yanchor='top',
            )

        # ── Stats bar (bottom) ────────────────────────────────────────────────
        stats_y = min_y_val - CARD_PAD - 92
        fig.add_shape(type='rect',
            x0=x_left, y0=stats_y - 24, x1=x_right, y1=stats_y + 24,
            fillcolor='rgba(255,255,255,0.03)',
            line=dict(color='rgba(255,255,255,0.09)', width=1),
        )
        fig.add_annotation(
            x=(x_left + x_right) / 2, y=stats_y,
            text=(
                f"<span style='color:#888'>Architecture  ·  </span>"
                f"<b style='color:#00D9FF'>{len(all_layers)}</b>"
                f"<span style='color:#666'> layers  ·  </span>"
                f"<b style='color:#D97BFF'>{total_params:,}</b>"
                f"<span style='color:#666'> trainable parameters</span>"
            ),
            showarrow=False,
            font=dict(size=12, color='rgba(200,200,200,0.8)', family='Courier New'),
            align='center',
        )

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            paper_bgcolor='#0D1117',
            plot_bgcolor='#0D1117',
            font=dict(family='Arial', color='white'),
            title=dict(
                text='<b>Neural Network Architecture</b>',
                font=dict(size=20, color='#E6EDF3', family='Arial'),
                x=0.5, xanchor='center', y=0.99, yanchor='top',
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(l=30, r=30, t=50, b=20),
            xaxis=dict(visible=False, range=[x_left - 10, x_right + 10]),
            yaxis=dict(visible=False, range=[stats_y - 36, max_y_val + CARD_PAD + 62]),
            height=720,
            hoverlabel=dict(
                bgcolor='#1C2333',
                font_size=12, font_family='Courier New', font_color='white',
                bordercolor='rgba(255,255,255,0.15)',
            ),
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
                                     mae_list: List[float]) -> go.Figure:
        """
        Create real-time updating metrics plot
        
        Args:
            epochs_list: List of epoch numbers
            loss_list: List of loss values
            mae_list: List of Mean Absolute Error values
            
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
        
        # Add MAE curve
        fig.add_trace(
            go.Scatter(
                x=epochs_list,
                y=mae_list,
                name='Mean Absolute Error',
                mode='lines+markers',
                line=dict(
                    color='#3498DB',
                    width=3,
                    shape='spline'
                ),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)',
                hovertemplate='<b>Epoch %{x}</b><br>MAE: %{y:.6f}<extra></extra>'
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
            title=dict(text='<b>Loss</b>', font=dict(size=14, color='#FF6B9D')),
            secondary_y=False,
            tickfont=dict(size=12, color='#FF6B9D')
        )
        
        fig.update_yaxes(
            title=dict(text='<b>Mean Absolute Error (MAE)</b>', font=dict(size=14, color='#3498DB')),
            secondary_y=True,
            tickfont=dict(size=12, color='#3498DB')
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
                                   mae_values: List[float]) -> go.Figure:
    """
    Create training metrics visualization
    
    Args:
        epochs: List of epoch numbers
        losses: List of loss values
        mae_values: List of Mean Absolute Error values
        
    Returns:
        Plotly figure with metrics
    """
    animator = TrainingAnimator(None, [], len(epochs))
    return animator.create_real_time_metrics_plot(epochs, losses, mae_values)


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
