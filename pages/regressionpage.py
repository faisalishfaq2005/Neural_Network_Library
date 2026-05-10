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

st.title("🧠 Neural Network Regression Trainer")
st.markdown("### Build, Train, and Evaluate a Neural Network with Intuitive Controls")

data=None
input_features=None
output_features=None
input_features_normalized=None
output_features_normalized=None
nn=LinkList()
nn_trained=LinkList()


st.markdown("#### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    input_features=data.iloc[:,:-1].values
    output_features=data.iloc[:,-1].values.reshape(-1,1)
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




if "nn_structure_regression" not in st.session_state:
    st.session_state.nn_structure_regression = []
if "nn_linklist_regression" not in st.session_state:
    st.session_state.nn_linklist_regression=[]
if "trained_model_layers_regression" not in st.session_state:
    st.session_state.trained_model_layers_regression=[]
if "prediction_regression" not in st.session_state:
    st.session_state.prediction_regression=[]


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
        
        st.session_state.nn_structure_regression.append({"type": "Activation", "activation": activation})

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
            
            # Calculate loss and accuracy
            error = output_array - predicted_output
            loss = mse_loss(output_array, predicted_output)
            epoch_loss += loss
            batch_count += 1
            
            # Calculate accuracy
            threshold = 0.05
            correct_predictions += np.sum(np.abs(output_array - predicted_output) <= threshold)
            total_predictions += output_array.size
            
            # Backward pass
            gradients_stack = nn.backward_propagation(error)
            nn.update_parameters(gradients_stack, learning_rate)
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # Store metrics
        epoch_metrics['epochs'].append(epoch_num + 1)
        epoch_metrics['losses'].append(avg_loss)
        epoch_metrics['accuracies'].append(accuracy)
        
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
                    "📈 Accuracy",
                    f"{accuracy:.2f}%",
                    delta=f"{accuracy:.2f}%" if epoch_num > 0 else None
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
    
    # Store trained layers
    layers = nn.get_all_nodes()
    for layer in layers:
        st.session_state.trained_model_layers_regression.append(layer)
    
    # Display final training summary
    st.markdown("### 📊 Training Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("🏆 Final Accuracy", f"{accuracy:.2f}%")
    with summary_col2:
        st.metric("📉 Final Loss", f"{epoch_metrics['losses'][-1]:.6f}")
    with summary_col3:
        st.metric("⏱️ Total Time", f"{time.time() - start_time:.1f}s")













st.markdown("### What Would You Like to Do Next?")
action = st.selectbox("Choose Action:", ["Check Model Accuracy", "Make Predictions"])

if action == "Check Model Accuracy":
    accuracy_percentage = 0
    if correct_predictions > 0 and total_predictions > 0:
        accuracy_percentage = (correct_predictions / total_predictions) * 100
    
    # Display accuracy with styled metric
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Model Accuracy", f"{accuracy_percentage:.2f}%")
    with col2:
        st.metric("✅ Correct Predictions", f"{correct_predictions}")
    with col3:
        st.metric("📊 Total Predictions", f"{total_predictions}")
    
elif action == "Make Predictions":
    for layer in st.session_state.trained_model_layers_regression:
        nn_trained.insert_node_without_updates(layer)
    
    st.markdown("### 🔮 Make New Predictions")
    st.info("Enter input features in the same format as your training data")
    
    user_input = st.text_area("Enter input features (comma-separated):", placeholder="e.g., 100,200,300")
    
    if st.button("Predict 🎯", key="predict_regression"):
        try:
            inputs = np.array([float(i) for i in user_input.split(",")]).reshape(1, -1)
            
            if inputs.size > 0:
                # Check if we have input metadata
                if hasattr(st.session_state, 'input_metadata'):
                    # Use new denormalization with metadata
                    inputs_normalized = (inputs - st.session_state.input_metadata['min_values']) / (
                        st.session_state.input_metadata['max_values'] - st.session_state.input_metadata['min_values']
                    )
                else:
                    # Fallback: calculate from normalized training data (old method)
                    input_min = np.min(input_features_normalized, axis=0)
                    input_max = np.max(input_features_normalized, axis=0)
                    inputs_normalized = (inputs - input_min) / (input_max - input_min)
                
                # Forward propagation
                predicted_output_normalized = nn_trained.forward_propogation(inputs_normalized)
                
                # Denormalize prediction
                if hasattr(st.session_state, 'output_metadata'):
                    prediction = denormalize(predicted_output_normalized, st.session_state.output_metadata)
                else:
                    # Fallback: use old method
                    output_min = np.min(output_features_normalized, axis=0)
                    output_max = np.max(output_features_normalized, axis=0)
                    prediction = predicted_output_normalized * (output_max - output_min) + output_min
                
                # Display prediction with nice formatting
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### Prediction Result")
                
                with col2:
                    st.metric("🎯 Predicted Value", f"{prediction.flatten()[0]:.4f}", 
                             delta=f"Based on {len(inputs[0])} features")
                
                st.success("Prediction completed successfully! ✅")
                
        except ValueError:
            st.error("❌ Please enter valid numbers separated by commas")
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
