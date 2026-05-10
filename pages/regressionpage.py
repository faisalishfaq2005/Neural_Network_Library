import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from DataStructures.linklist import LinkList
from DataStructures.queueADT import Queue
from Layers.ActivationLayer import ActivationLayer
from Layers.DenseLayer import DenseLayer
from Layers.InputLayer import InputLayer
from Layers.ActivationFunctions import sigmoid,sigmoid_derivative,relu,relu_derivative,leaky_relu,leaky_relu_derivative
from Supporting_Functions.normalization import normalize,denormalize
from Supporting_Functions.visualization import (
    NetworkVisualizer, TrainingAnimator, visualize_network_architecture,
    create_training_metrics_display
)
from Layers.lossFunctions import binary_cross_entropy,mse_loss
from Supporting_Functions.batching import batch_split1,Batch
import plotly.graph_objects as go

st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .btn-primary {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 18px;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .btn-primary:hover {
        transform: scale(1.1);
        background-color: #45a049;
    }
    .animate-layer {
        transition: transform 0.3s, background-color 0.3s;
        margin: 10px;
        padding: 15px;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
        font-size: 18px;
        cursor: pointer;
    }
    .animate-layer:hover {
        transform: scale(1.1);
        background-color: rgba(76, 175, 80, 0.3);
    }
    .loss-box {
        background-color: rgba(255, 193, 7, 0.1);
        padding: 15px;
        border: 2px dashed #FFC107;
        margin: 10px;
        text-align: center;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Reset all state when navigating here from a different page ───────────────
_REGRESSION_KEYS = [
    'nn_structure_regression', 'nn_linklist_regression',
    'trained_model_layers_regression', 'prediction_regression',
    'nn_trained_regression', 'model_mae_regression',
    'model_trained_regression', 'csv_columns', 'original_data',
    'col_is_categorical_regression', 'col_categories_regression',
    'input_metadata', 'output_metadata', 'target_column',
]
if st.session_state.get('active_page') != 'regression':
    for _k in _REGRESSION_KEYS:
        st.session_state.pop(_k, None)
    st.session_state['active_page'] = 'regression'

# Initialize session state variables at the very beginning
if "nn_structure_regression" not in st.session_state:
    st.session_state.nn_structure_regression = []
if "nn_linklist_regression" not in st.session_state:
    st.session_state.nn_linklist_regression=[]
if "trained_model_layers_regression" not in st.session_state:
    st.session_state.trained_model_layers_regression=[]
if "prediction_regression" not in st.session_state:
    st.session_state.prediction_regression=[]
if "nn_trained_regression" not in st.session_state:
    st.session_state.nn_trained_regression = None
if "model_mae_regression" not in st.session_state:
    st.session_state.model_mae_regression = None
if "model_trained_regression" not in st.session_state:
    st.session_state.model_trained_regression = False
if "csv_columns" not in st.session_state:
    st.session_state.csv_columns = None
if "original_data" not in st.session_state:
    st.session_state.original_data = None
if "col_is_categorical_regression" not in st.session_state:
    st.session_state.col_is_categorical_regression = {}
if "col_categories_regression" not in st.session_state:
    st.session_state.col_categories_regression = {}

st.title("🧠 Neural Network Regression Trainer")
st.markdown("### Build, Train, and Evaluate a Neural Network with Intuitive Controls")

data=None
input_features=None
output_features=None
input_features_normalized=None
output_features_normalized=None
nn=LinkList()
nn_trained=LinkList()


def _reset_regression_layers():
    """Called by the file uploader on_change — wipes layer/model state for a clean start."""
    for _k in ['nn_structure_regression', 'nn_linklist_regression',
               'trained_model_layers_regression', 'prediction_regression',
               'nn_trained_regression', 'model_mae_regression', 'model_trained_regression',
               'csv_columns', 'original_data', 'col_is_categorical_regression',
               'col_categories_regression', 'input_metadata', 'output_metadata', 'target_column']:
        st.session_state.pop(_k, None)

st.markdown("#### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"],
                                  key="reg_file_uploader",
                                  on_change=_reset_regression_layers)
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    input_features=data.iloc[:,:-1].values
    output_features=data.iloc[:,-1].values.reshape(-1,1)
    
    # Store column names and original data for predictions
    st.session_state.csv_columns = list(data.columns[:-1])
    st.session_state.original_data = data
    st.session_state.target_column = data.columns[-1]

    # Detect categorical vs numeric columns for prediction UI and encoding
    col_is_categorical = {}
    col_categories = {}
    for col_name in data.columns[:-1]:
        col_data = data[col_name]
        numeric_check = pd.to_numeric(col_data, errors='coerce')
        if numeric_check.isna().sum() / max(len(col_data), 1) > 0.3:
            col_is_categorical[col_name] = True
            col_categories[col_name] = np.unique(col_data.dropna().values).tolist()
        else:
            col_is_categorical[col_name] = False
    st.session_state.col_is_categorical_regression = col_is_categorical
    st.session_state.col_categories_regression = col_categories
    
    if input_features is not None:
        print(input_features.shape)

    st.write("### Uploaded Data", data.head())
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

st.markdown("#### Step 2: Data Normalization")
normalize_data = st.checkbox("Normalize Data?", value=True)
if normalize_data and input_features is not None and input_features.size > 0 and output_features is not None and output_features.size > 0:
    try:
        # New normalize function returns (data, metadata, encoded_categorical)
        input_features_normalized, input_metadata, _ = normalize(
            input_features, 
            handle_categorical=True,
            categorical_method='label'
        )
        
        output_features_normalized, output_metadata, _ = normalize(
            output_features,
            handle_categorical=True,
            categorical_method='label'
        )
        
        # Store metadata for denormalization later
        st.session_state.input_metadata = input_metadata
        st.session_state.output_metadata = output_metadata

        st.success("Data normalized successfully!")
        st.write("Normalized Data Preview:")
        st.write(f"Input shape: {input_features_normalized.shape}")
        st.write(f"Output shape: {output_features_normalized.shape}")
    except Exception as e:
        st.error(f"Error during normalization: {str(e)}")
        st.info("Please ensure your dataset contains valid numeric values or categorical values that can be encoded.")
        st.stop()
else:
    st.warning("Data will not be normalized.")
    input_features_normalized = input_features
    output_features_normalized = output_features

print("Input Features Normalized Shape:", input_features_normalized.shape if input_features_normalized is not None else "None")
print("Output Features Normalized Shape:", output_features_normalized.shape if output_features_normalized is not None else "None")



st.title("Neural Network Builder")



layer_type = st.selectbox("Select Layer Type:", ["Dense Layer", "Activation Layer"])

if layer_type=="Dense Layer":
    neurons = int(st.number_input("Number of Neurons:"))
elif layer_type=="Activation Layer":
    activation = st.selectbox("Activation Function:", ["ReLU", "Sigmoid"])



if st.button("Add Layer"):
    
    if layer_type == "Dense Layer":
        if nn.is_empty()==True:
            
            layer=(DenseLayer(len(input_features_normalized[0]),neurons))
            st.session_state.nn_linklist_regression.append(layer)
            
            st.session_state.nn_structure_regression.append({"type": "Dense", "neurons": neurons})
        else:
           
            layer=(DenseLayer(None,neurons))
            st.session_state.nn_linklist_regression.append(layer)
            st.session_state.nn_structure_regression.append({"type": "Dense", "neurons": neurons})
    elif layer_type == "Activation Layer":
        activation_functions={"ReLU":(relu,relu_derivative),"Sigmoid":(sigmoid,sigmoid_derivative)}
        
        activation_data=activation_functions.get(activation)
        if activation_data:
            act_function=activation_data[0]
            act_derivative=activation_data[1]
        layer=(ActivationLayer(act_function,act_derivative))
        st.session_state.nn_linklist_regression.append(layer)
        
        # Get neuron count from last dense layer
        last_neuron_count = None
        for i in range(len(st.session_state.nn_structure_regression) - 1, -1, -1):
            if st.session_state.nn_structure_regression[i]["type"] == "Dense":
                last_neuron_count = st.session_state.nn_structure_regression[i]["neurons"]
                break
        
        st.session_state.nn_structure_regression.append({
            "type": "Activation", 
            "activation": activation,
            "neurons": last_neuron_count
        })

st.markdown("### Current Neural Network Structure:")
if st.session_state.nn_structure_regression:
    for idx, layer in enumerate(st.session_state.nn_structure_regression):
        if layer["type"] == "Dense":
            st.markdown(f"**Layer {idx + 1}: Dense Layer ({layer['neurons']} neurons)**")
        elif layer["type"] == "Activation":
            st.markdown(f"**Layer {idx + 1}: Activation Layer ({layer['activation']})**")
else:
    st.warning("No layers added yet.")

if st.session_state.nn_structure_regression:
    st.markdown("### 🎨 Network Architecture Visualization")
    try:
        visualizer = NetworkVisualizer(st.session_state.nn_structure_regression, len(input_features_normalized[0]))
        fig = visualizer.visualize_architecture(show_layer_info=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not visualize network: {str(e)}")


st.markdown("#### Step 4: Training Configuration")
batch_size = int(st.number_input("Batch Size:", value=32, min_value=1, step=1))
epochs = int(st.number_input("Number of Epochs:", value=1000, min_value=1, step=1))
learning_rate = st.number_input("Learning Rate:", value=0.01, min_value=0.0001, step=0.001, format="%.4f")



st.markdown("#### Step 5: Train Your Neural Network")
correct_predictions = 0
total_predictions = 0

if st.button("Start Training 🚀", key="train_regression"):
    if st.session_state.nn_linklist_regression:
        for layer in st.session_state.nn_linklist_regression:
            nn.insert_node(layer)

    # Initialize metrics tracking
    epoch_metrics = {
        'epochs': [],
        'losses': [],
        'accuracies': [],
    }
    
    # Create beautiful training visualization layout
    st.markdown("### 🎯 Training Dashboard")
    
    # Create metrics display placeholders
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        epoch_placeholder = st.empty()
    with metric_col2:
        loss_placeholder = st.empty()
    with metric_col3:
        accuracy_placeholder = st.empty()
    with metric_col4:
        time_placeholder = st.empty()
    
    # Create chart placeholder
    chart_placeholder = st.empty()
    
    # Create data flow animation placeholder
    animation_placeholder = st.empty()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    for epoch_num in range(epochs):
        epoch_start = time.time()
        
        batch_queue = batch_split1(input_features_normalized, output_features_normalized, batch_size)
        epoch_loss = 0
        batch_count = 0
        
        while batch_queue.is_empty() != True:
            batch = batch_queue.dequeue()
            input_array = batch.feature_array
            output_array = batch.output_array
            
            # Forward pass
            predicted_output = nn.forward_propogation(input_array)
            
            if epoch_num == epochs - 1:
                st.session_state.prediction_regression.append(predicted_output)
            
            # Calculate loss and MAE
            error = output_array - predicted_output
            loss = mse_loss(output_array, predicted_output)
            epoch_loss += loss
            batch_count += 1
            
            # Calculate Mean Absolute Error (MAE)
            correct_predictions += np.sum(np.abs(output_array - predicted_output))
            total_predictions += output_array.size
            
            # Backward pass
            gradients_stack = nn.backward_propagation(error)
            nn.update_parameters(gradients_stack, learning_rate)
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        mae = (correct_predictions / total_predictions) if total_predictions > 0 else 0
        
        # Store metrics
        epoch_metrics['epochs'].append(epoch_num + 1)
        epoch_metrics['losses'].append(avg_loss)
        epoch_metrics['accuracies'].append(mae)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Update progress bar
        progress = min((epoch_num + 1) / epochs, 1.0)
        progress_bar.progress(progress)
        
        # Update metrics display every 10 epochs or on last epoch
        if (epoch_num + 1) % max(1, epochs // 10) == 0 or epoch_num == epochs - 1:
            with epoch_placeholder.container():
                st.metric(
                    "📊 Epoch",
                    f"{epoch_num + 1}/{epochs}",
                    delta=f"{progress*100:.1f}%"
                )
            
            with loss_placeholder.container():
                st.metric(
                    "📉 Loss",
                    f"{avg_loss:.6f}",
                    delta=f"{avg_loss:.6f}" if epoch_num > 0 else None,
                    delta_color="inverse"
                )
            
            with accuracy_placeholder.container():
                st.metric(
                    "� MAE (Mean Absolute Error)",
                    f"{mae:.6f}",
                    delta=f"{mae:.6f}" if epoch_num > 0 else None,
                    delta_color="inverse"
                )
            
            with time_placeholder.container():
                st.metric(
                    "⏱️ Time",
                    f"{total_time:.1f}s",
                    delta=f"{epoch_time:.2f}s/epoch"
                )
            
            # Update metrics chart
            with chart_placeholder.container():
                try:
                    fig = create_training_metrics_display(
                        epoch_metrics['epochs'],
                        epoch_metrics['losses'],
                        epoch_metrics['accuracies']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display metrics: {str(e)}")
        
        # Update status
        status_text.info(f"⏳ Training epoch {epoch_num + 1}/{epochs}... ({progress*100:.1f}%)")
    
    # Final status
    status_text.success("✅ Model trained successfully! 🎉")
    progress_bar.progress(1.0)
    
    # Store trained layers and model info in session state
    layers = nn.get_all_nodes()
    for layer in layers:
        st.session_state.trained_model_layers_regression.append(layer)
    
    # Persist the trained model to session state for predictions
    st.session_state.nn_trained_regression = nn
    st.session_state.model_mae_regression = mae
    st.session_state.model_trained_regression = True
    
    # Display final training summary
    st.markdown("### 📊 Training Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("📏 Final MAE", f"{mae:.6f}")
    with summary_col2:
        st.metric("📉 Final Loss", f"{epoch_metrics['losses'][-1]:.6f}")
    with summary_col3:
        st.metric("⏱️ Total Time", f"{time.time() - start_time:.1f}s")













# Show model trained status and options for what to do next
if st.session_state.model_trained_regression:
    st.markdown("### ✅ Model Status: TRAINED")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("📏 Model MAE", f"{st.session_state.model_mae_regression:.6f}")
    with status_col2:
        st.metric("🧠 Total Layers", len(st.session_state.nn_structure_regression))
    with status_col3:
        st.metric("✓ Status", "Ready to Predict")

st.markdown("### What Would You Like to Do Next?")
if st.session_state.model_trained_regression:
    action = st.selectbox("Choose Action:", ["Check Model Performance", "Make Predictions"])
    
    if action == "Check Model Performance":
        # Display training summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📏 Final MAE", f"{st.session_state.model_mae_regression:.6f}")
        with col2:
            st.metric("🧠 Network Layers", len(st.session_state.nn_structure_regression))
        with col3:
            st.metric("✓ Model Status", "Trained")
        
    elif action == "Make Predictions":
        st.markdown("### 🔮 Make New Predictions")
        st.info("Enter values for each input feature")

        if st.session_state.csv_columns and st.session_state.original_data is not None:
            prediction_inputs = {}
            col_is_cat = st.session_state.col_is_categorical_regression
            col_cats = st.session_state.col_categories_regression
            input_cols = st.columns(min(3, len(st.session_state.csv_columns)))

            for idx, col_name in enumerate(st.session_state.csv_columns):
                col_index = idx % len(input_cols)
                with input_cols[col_index]:
                    if col_is_cat.get(col_name, False):
                        unique_vals = col_cats.get(col_name, list(st.session_state.original_data[col_name].unique()))
                        prediction_inputs[col_name] = st.selectbox(
                            f"📋 {col_name}",
                            unique_vals,
                            key=f"select_{col_name}"
                        )
                    else:
                        prediction_inputs[col_name] = st.number_input(
                            f"🔢 {col_name}",
                            value=0.0,
                            step=0.01,
                            format="%.4f",
                            key=f"input_{col_name}"
                        )

            if st.button("Predict 🎯", key="predict_regression"):
                try:
                    input_metadata = getattr(st.session_state, 'input_metadata', None)

                    # Build normalized input matching training encoding exactly
                    input_normalized = np.zeros((1, len(st.session_state.csv_columns)), dtype=float)
                    numeric_col_counter = 0

                    for col_idx, col_name in enumerate(st.session_state.csv_columns):
                        val = prediction_inputs[col_name]
                        if col_is_cat.get(col_name, False):
                            # Label-encode matching np.unique sort order used during training
                            categories = [str(c) for c in col_cats.get(col_name, [])]
                            val_str = str(val)
                            encoded = categories.index(val_str) if val_str in categories else 0
                            input_normalized[0][col_idx] = float(encoded)
                        else:
                            float_val = float(val)
                            if (input_metadata is not None and
                                    input_metadata.get('min_values') is not None and
                                    numeric_col_counter < len(input_metadata['min_values'])):
                                min_v = float(input_metadata['min_values'][numeric_col_counter])
                                max_v = float(input_metadata['max_values'][numeric_col_counter])
                                diff = (max_v - min_v) if (max_v - min_v) != 0 else 1.0
                                input_normalized[0][col_idx] = (float_val - min_v) / diff
                            else:
                                input_normalized[0][col_idx] = float_val
                            numeric_col_counter += 1

                    if st.session_state.nn_trained_regression is not None:
                        prediction = st.session_state.nn_trained_regression.forward_propogation(input_normalized)

                        if prediction is not None:
                            out_meta = getattr(st.session_state, 'output_metadata', None)
                            if out_meta is not None and out_meta.get('min_values') is not None:
                                denormalized_pred = denormalize(prediction, out_meta)
                            else:
                                denormalized_pred = prediction

                            st.success("✅ Prediction Successful!")
                            st.markdown("### 🎯 Prediction Result")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Value", f"{float(denormalized_pred.flatten()[0]):.4f}")
                            with col2:
                                st.metric("Normalized Value", f"{float(prediction.flatten()[0]):.4f}")
                        else:
                            st.error("Could not generate prediction.")
                    else:
                        st.error("Model not available. Please train the model first.")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Please upload a CSV file first to proceed with predictions.")
else:
    st.warning("⚠️ Please train the model first before making predictions.")
