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


# CSS for frontend styling
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
_CLASSIFICATION_KEYS = [
    'nn_structure_classification', 'nn_linklist_classification',
    'trained_model_layers_classification', 'final_loss_classification',
    'nn_trained_classification', 'model_accuracy_classification',
    'model_trained_classification', 'csv_columns_classification',
    'original_data_classification', 'col_is_categorical_classification',
    'col_categories_classification', 'input_metadata',
    'output_features_original', 'target_column_classification',
]
if st.session_state.get('active_page') != 'classification':
    for _k in _CLASSIFICATION_KEYS:
        st.session_state.pop(_k, None)
    st.session_state['active_page'] = 'classification'

# Initialize session state variables at the very beginning
if "nn_structure_classification" not in st.session_state:
    st.session_state.nn_structure_classification = []
if "nn_linklist_classification" not in st.session_state:
    st.session_state.nn_linklist_classification=[]
if "trained_model_layers_classification" not in st.session_state:
    st.session_state.trained_model_layers_classification=[]
if "final_loss_classification" not in st.session_state:
    st.session_state.final_loss_classification=[]
if "nn_trained_classification" not in st.session_state:
    st.session_state.nn_trained_classification = None
if "model_accuracy_classification" not in st.session_state:
    st.session_state.model_accuracy_classification = None
if "model_trained_classification" not in st.session_state:
    st.session_state.model_trained_classification = False
if "csv_columns_classification" not in st.session_state:
    st.session_state.csv_columns_classification = None
if "original_data_classification" not in st.session_state:
    st.session_state.original_data_classification = None
if "col_is_categorical_classification" not in st.session_state:
    st.session_state.col_is_categorical_classification = {}
if "col_categories_classification" not in st.session_state:
    st.session_state.col_categories_classification = {}

st.title("🧠 Neural Network Classification Trainer")
st.markdown("### Build, Train, and Evaluate a Neural Network with Intuitive Controls")

data=None
input_features=None
output_features=None
input_features_normalized=None

nn=LinkList()
nn_trained=LinkList()

st.markdown("#### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    input_features=data.iloc[:,:-1].values
    output_features=data.iloc[:,-1].values.reshape(-1,1)
    
    # Store column names and original data for predictions
    st.session_state.csv_columns_classification = list(data.columns[:-1])
    st.session_state.original_data_classification = data
    st.session_state.target_column_classification = data.columns[-1]

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
    st.session_state.col_is_categorical_classification = col_is_categorical
    st.session_state.col_categories_classification = col_categories
    
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
        
        # Store metadata for denormalization later
        st.session_state.input_metadata = input_metadata
        st.session_state.output_features_original = output_features

        st.success("Data normalized successfully!")
        st.write("Normalized Data Preview:")
        st.write(f"Input shape: {input_features_normalized.shape}")
    except Exception as e:
        st.error(f"Error during normalization: {str(e)}")
        st.info("Please ensure your dataset contains valid numeric values or categorical values that can be encoded.")
        st.stop()
else:
    st.warning("Data will not be normalized.")
    input_features_normalized = input_features

print("Input Features Normalized Shape:", input_features_normalized.shape if input_features_normalized is not None else "None")





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
            st.session_state.nn_linklist_classification.append(layer)
            
            st.session_state.nn_structure_classification.append({"type": "Dense", "neurons": neurons})
        else:
           
            layer=(DenseLayer(None,neurons))
            st.session_state.nn_linklist_classification.append(layer)
            st.session_state.nn_structure_classification.append({"type": "Dense", "neurons": neurons})
    elif layer_type == "Activation Layer":
        activation_functions={"ReLU":(relu,relu_derivative),"Sigmoid":(sigmoid,sigmoid_derivative)}
        
        activation_data=activation_functions.get(activation)
        if activation_data:
            act_function=activation_data[0]
            act_derivative=activation_data[1]
        layer=(ActivationLayer(act_function,act_derivative))
        st.session_state.nn_linklist_classification.append(layer)
        
        # Get neuron count from last dense layer
        last_neuron_count = None
        for i in range(len(st.session_state.nn_structure_classification) - 1, -1, -1):
            if st.session_state.nn_structure_classification[i]["type"] == "Dense":
                last_neuron_count = st.session_state.nn_structure_classification[i]["neurons"]
                break
        
        st.session_state.nn_structure_classification.append({
            "type": "Activation", 
            "activation": activation,
            "neurons": last_neuron_count
        })

st.markdown("### Current Neural Network Structure:")
if st.session_state.nn_structure_classification:
    for idx, layer in enumerate(st.session_state.nn_structure_classification):
        if layer["type"] == "Dense":
            st.markdown(f"**Layer {idx + 1}: Dense Layer ({layer['neurons']} neurons)**")
        elif layer["type"] == "Activation":
            st.markdown(f"**Layer {idx + 1}: Activation Layer ({layer['activation']})**")
else:
    st.warning("No layers added yet.")

if st.session_state.nn_structure_classification:
    st.markdown("### 🎨 Network Architecture Visualization")
    try:
        visualizer = NetworkVisualizer(st.session_state.nn_structure_classification, len(input_features_normalized[0]))
        fig = visualizer.visualize_architecture(show_layer_info=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not visualize network: {str(e)}")

st.markdown("#### Step 4: Training Configuration")
batch_size = int(st.number_input("Batch Size:", value=32, min_value=1, step=1))
epochs = int(st.number_input("Number of Epochs:", value=1000, min_value=1, step=1))
learning_rate = st.number_input("Learning Rate:", value=0.01, min_value=0.0001, step=0.001, format="%.4f")


st.markdown("#### Step 5: Train Your Neural Network")

if st.button("Start Training 🚀", key="train_classification"):
    if st.session_state.nn_linklist_classification:
        for layer in st.session_state.nn_linklist_classification:
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
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    total_correct = 0
    total_samples = 0
    
    for epoch_num in range(epochs):
        epoch_start = time.time()
        
        batch_queue = batch_split1(input_features_normalized, output_features, batch_size)
        epoch_loss = 0
        batch_count = 0
        epoch_correct = 0
        epoch_total = 0
        
        while batch_queue.is_empty() != True:
            batch = batch_queue.dequeue()
            input_array = batch.feature_array
            output_array = batch.output_array
            
            # Forward pass
            predicted_output = nn.forward_propogation(input_array)
            
            # Calculate loss
            loss = binary_cross_entropy(output_array, predicted_output)
            epoch_loss += loss
            batch_count += 1
            
            if epoch_num == epochs - 1:
                st.session_state.final_loss_classification.append(loss)
            
            # Calculate accuracy (for classification)
            predictions = (predicted_output > 0.5).astype(int)
            correct = np.sum(predictions == output_array)
            epoch_correct += correct
            epoch_total += output_array.size
            total_correct += correct
            total_samples += output_array.size
            
            # Backward pass
            error = output_array - predicted_output
            gradients_stack = nn.backward_propagation(error)
            nn.update_parameters(gradients_stack, learning_rate)
        
        # Calculate average loss and accuracy for epoch
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        accuracy = (epoch_correct / epoch_total * 100) if epoch_total > 0 else 0
        
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
                    "📉 Loss (BCE)",
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
    
    # Store trained layers and model info in session state
    layers = nn.get_all_nodes()
    for layer in layers:
        st.session_state.trained_model_layers_classification.append(layer)
    
    # Persist the trained model to session state for predictions
    final_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    st.session_state.nn_trained_classification = nn
    st.session_state.model_accuracy_classification = final_accuracy
    st.session_state.model_trained_classification = True
    
    # Display final training summary
    st.markdown("### 📊 Training Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("🏆 Final Accuracy", f"{final_accuracy:.2f}%")
    with summary_col2:
        st.metric("📉 Final Loss", f"{epoch_metrics['losses'][-1]:.6f}")
    with summary_col3:
        st.metric("⏱️ Total Time", f"{time.time() - start_time:.1f}s")





# Show model trained status and options for what to do next
if st.session_state.model_trained_classification:
    st.markdown("### ✅ Model Status: TRAINED")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("🏆 Model Accuracy", f"{st.session_state.model_accuracy_classification:.2f}%")
    with status_col2:
        st.metric("🧠 Total Layers", len(st.session_state.nn_structure_classification))
    with status_col3:
        st.metric("✓ Status", "Ready to Predict")

st.markdown("### What Would You Like to Do Next?")
if st.session_state.model_trained_classification:
    action = st.selectbox("Choose Action:", ["Check Model Performance", "Make Predictions"])
    
    if action == "Check Model Performance":
        # Display training summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Final Accuracy", f"{st.session_state.model_accuracy_classification:.2f}%")
        with col2:
            st.metric("🧠 Network Layers", len(st.session_state.nn_structure_classification))
        with col3:
            st.metric("✓ Model Status", "Trained")
        
    elif action == "Make Predictions":
        st.markdown("### 🔮 Make New Predictions")
        st.info("Enter values for each input feature")

        if st.session_state.csv_columns_classification and st.session_state.original_data_classification is not None:
            prediction_inputs = {}
            col_is_cat = st.session_state.col_is_categorical_classification
            col_cats = st.session_state.col_categories_classification
            input_cols = st.columns(min(3, len(st.session_state.csv_columns_classification)))

            for idx, col_name in enumerate(st.session_state.csv_columns_classification):
                col_index = idx % len(input_cols)
                with input_cols[col_index]:
                    if col_is_cat.get(col_name, False):
                        unique_vals = col_cats.get(col_name, list(st.session_state.original_data_classification[col_name].unique()))
                        prediction_inputs[col_name] = st.selectbox(
                            f"📋 {col_name}",
                            unique_vals,
                            key=f"select_{col_name}_clf"
                        )
                    else:
                        prediction_inputs[col_name] = st.number_input(
                            f"🔢 {col_name}",
                            value=0.0,
                            step=0.01,
                            format="%.4f",
                            key=f"input_{col_name}_clf"
                        )

            if st.button("Predict 🎯", key="predict_classification"):
                try:
                    input_metadata = getattr(st.session_state, 'input_metadata', None)

                    # Build normalized input matching training encoding exactly
                    input_normalized = np.zeros((1, len(st.session_state.csv_columns_classification)), dtype=float)
                    numeric_col_counter = 0

                    for col_idx, col_name in enumerate(st.session_state.csv_columns_classification):
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

                    if st.session_state.nn_trained_classification is not None:
                        prediction_probability = st.session_state.nn_trained_classification.forward_propogation(input_normalized)
                        prediction_probability = float(prediction_probability.flatten()[0])
                        predicted_class = 1 if prediction_probability >= 0.5 else 0

                        st.success("✅ Prediction Successful!")
                        st.markdown("### 🎯 Prediction Result")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("### Predicted Class")
                        with col2:
                            st.metric("🎯 Class", f"{predicted_class}",
                                      delta=f"Confidence: {prediction_probability:.2%}")

                        st.markdown("#### Probability Distribution")
                        prob_col1, prob_col2 = st.columns(2)
                        with prob_col1:
                            st.write(f"**Class 0**: {(1 - prediction_probability):.2%}")
                            st.progress(1 - prediction_probability)
                        with prob_col2:
                            st.write(f"**Class 1**: {prediction_probability:.2%}")
                            st.progress(prediction_probability)
                    else:
                        st.error("Model not available. Please train the model first.")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Please upload a CSV file first to proceed with predictions.")
else:
    st.warning("⚠️ Please train the model first before making predictions.")



        
