import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import os
import SimpleITK as sitk
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Configuration
MODEL_PATH = "your_model.pth"  # Update this with your actual model path
SAMPLE_SCANS_DIR = "sample_scans"  # Directory containing sample CT scans
TENSORBOARD_LOGS_DIR = "runs"  # Directory containing tensorboard logs

class CTScanPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def preprocess_ct_scan(self, mhd_path, raw_path):
        try:
            # Read the MHD file to get the image
            if mhd_path.endswith('.mhd'):
                image = sitk.ReadImage(mhd_path)
            else:
                image = sitk.ReadImage(mhd_path)
            
            image_array = sitk.GetArrayFromImage(image)
            
            image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).to(self.device)
            
            return image_tensor, image_array
        except Exception as e:
            print(f"Error preprocessing CT scan: {e}")
            return None, None
    
    def predict_classification(self, image_tensor):
        if self.model is None:
            return "Model not loaded", 0.0
        
        try:
            with torch.no_grad():
                output = self.model(image_tensor)
                if isinstance(output, tuple):
                    classification_output = output[0] 
                else:
                    classification_output = output
                
                probabilities = torch.softmax(classification_output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Map class index to class name (adjust based on your classes)
                class_names = ["Benign", "Malignant"]  # Update with your actual classes
                predicted_label = class_names[predicted_class]
                
                return predicted_label, confidence
        except Exception as e:
            print(f"Error in classification: {e}")
            return "Error", 0.0
    
    def predict_segmentation(self, image_tensor):
        """Perform segmentation prediction"""
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                # Assuming your model returns segmentation mask
                output = self.model(image_tensor)
                if isinstance(output, tuple):
                    segmentation_output = output[1]  # Assuming segmentation is second output
                else:
                    segmentation_output = output
                
                # Apply sigmoid/softmax based on your model output
                segmentation_mask = torch.sigmoid(segmentation_output)
                segmentation_mask = (segmentation_mask > 0.5).float()
                
                return segmentation_mask.cpu().numpy()
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None

def create_ct_visualization(array, segmentation_mask=None):
    if array is None:
        return None
    
    fig, axes=plt.subplots(2,3, figsize=(15,10))
    idx=[0,20,40,60,80,100,120]
    for i in range(6):
        axes[i//3, i%3].imshow(array[idx[i],:,:], cmap='gray', vmin=-200, vmax=200)

    plt.tight_layout()
    
    return fig

def create_training_graphs():
    """Create sample training graphs (replace with actual tensorboard data)"""
    # Sample data - replace with actual tensorboard log parsing
    epochs = list(range(1, 51))
    train_loss = [0.8 - 0.01 * i + 0.05 * np.sin(i/5) for i in epochs]
    val_loss = [0.85 - 0.008 * i + 0.08 * np.sin(i/4) for i in epochs]
    train_acc = [0.6 + 0.008 * i - 0.02 * np.sin(i/3) for i in epochs]
    val_acc = [0.55 + 0.007 * i - 0.03 * np.sin(i/4) for i in epochs]
    
    # Create loss plot
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss'))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))
    fig_loss.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
    
    # Create accuracy plot
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines', name='Training Accuracy'))
    fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation Accuracy'))
    fig_acc.update_layout(title='Training and Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
    
    return fig_loss, fig_acc

def process_uploaded_files(mhd_file, raw_file=None):
    if mhd_file is None:
        return "Please upload an MHD file", None, None, None, None
    
    try:
        predictor = CTScanPredictor(MODEL_PATH)
        
        image_tensor, image_array = predictor.preprocess_ct_scan(mhd_file.name, raw_file.name if raw_file else None)
        
        if image_tensor is None:
            return "Error processing CT scan", None, None, None, None
        
        # Perform classification
        class_prediction, confidence = predictor.predict_classification(image_tensor)
        
        segmentation_mask = predictor.predict_segmentation(image_tensor)
        
        ct_viz = create_ct_visualization(image_array, segmentation_mask)
        
        # Create results text
        results_text = f"""
        <div style="padding: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h3>Prediction Results</h3>
            <p><strong>Classification:</strong> {class_prediction}</p>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Segmentation:</strong> {'Completed' if segmentation_mask is not None else 'Failed'}</p>
        </div>
        """
        
        return results_text, ct_viz, ct_viz, class_prediction, f"{confidence:.2%}"
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

def process_sample_scan(sample_path):
    if not os.path.exists(sample_path):
        return "Sample file not found", None, None, None, None
    
    return process_uploaded_files(type('obj', (object,), {'name': sample_path})())

def create_interface():
    
    with gr.Blocks(theme=gr.themes.Soft(), title="CT Scan Analysis - Deep Learning") as demo:
        gr.Markdown("""
        # CT Scan Classification and Segmentation
        
        This application uses deep learning models trained with PyTorch to perform classification and segmentation on CT scans.
        Upload your MHD/RAW files or try the sample scans below.
        """)
        
    
        with gr.Tabs():
            with gr.TabItem("CLASSIFICATION"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload CT Scan Files")
                        mhd_file = gr.File(label="Upload MHD file", file_types=[".mhd"])
                        raw_file = gr.File(label="Upload RAW file (optional)", file_types=[".raw"])
                        predict_btn = gr.Button("Predict", variant="primary")
                        
                        gr.Markdown("### Sample CT Scans")
                        gr.Markdown("Click on any sample below to run prediction:")
                        
                        # Sample scan buttons (you'll need to add actual sample files)
                        sample_buttons = []
                        for i in range(1, 5):
                            btn = gr.Button(f"Sample Scan {i}", variant="secondary")
                            sample_buttons.append(btn)
                    
                    with gr.Column(scale=3):
                        results_display = gr.HTML(label="Results")
                        
                        with gr.Row():
                            classification_viz = gr.Plot(label="Classification Visualization")
                            #segmentation_viz = gr.Plot(label="Segmentation Visualization")
                        with gr.Row():
                            scroll_viz=gr.Plot(label="Scroll through CT Scan")

            
            # Training information tab 
            with gr.TabItem("Training Information"):
                gr.Markdown("""
                ## Model Training Information
                
                This model was trained using the Manning Deep Learning with PyTorch book methodology.
                
                ### Model Architecture
                - **Task**: Classification and Segmentation
                - **Framework**: PyTorch
                - **Input**: CT Scan images (MHD/RAW format)
                - **Output**: Binary classification + segmentation mask
                
                ### Training Details
                - **Dataset**: Custom CT scan dataset
                - **Epochs**: 50
                - **Batch Size**: 16
                - **Optimizer**: Adam
                - **Loss Function**: Combined classification and segmentation loss
                """)
                
                # Training graphs
                loss_fig, acc_fig = create_training_graphs()
                
                with gr.Row():
                    gr.Plot(value=loss_fig, label="Loss Curves")
                    gr.Plot(value=acc_fig, label="Accuracy Curves")
            
            # About tab
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About This Application
                
                This application demonstrates deep learning capabilities for medical image analysis,
                specifically focused on CT scan classification and segmentation.
                
                ### Features
                - **Classification**: Determines the category of the CT scan
                - **Segmentation**: Identifies and masks regions of interest
                - **Visualization**: Interactive CT scan viewing with overlays
                - **Sample Data**: Pre-loaded samples for testing
                
                ### Technical Stack
                - **Backend**: PyTorch for deep learning inference
                - **Frontend**: Gradio for web interface
                - **Visualization**: Plotly for interactive plots
                - **Image Processing**: SimpleITK for medical image handling
                
                ### Usage Instructions
                1. Upload MHD file (and optionally RAW file)
                2. Click "Predict" to run the model
                3. View results in the visualization panels
                4. Alternatively, try the sample scans for quick testing
                """)
        
        # Event handlers
        predict_btn.click(
            fn=process_uploaded_files,
            inputs=[mhd_file, raw_file],
            outputs=[results_display, classification_viz, gr.Textbox(visible=False), gr.Textbox(visible=False)]
        )
        
        # Sample scan event handlers
        for i, btn in enumerate(sample_buttons):
            sample_path = f"sample_scans/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
            btn.click(
                fn=lambda path=sample_path: process_sample_scan(path),
                inputs=[],
                outputs=[results_display, classification_viz, gr.Textbox(visible=False), gr.Textbox(visible=False)]
            )
    
    return demo

# Main execution
if __name__ == "__main__":
    # Create the interface
    demo = create_interface()
    
    demo.launch()
    # Launch the application
    #demo.launch(
    #    server_name="0.0.0.0",
    #    server_port=7860,
    #    share=False,
    #    debug=True
    #)