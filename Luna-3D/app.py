import gradio as gr
import torch
import numpy as np
import SimpleITK as sitk
import os
import tempfile
import shutil
import re
import matplotlib.pyplot as plt

from segmentation.model import UNetWrapper
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "seg_2025-08-05_10.19.19_final-cls.best.state" 
CONTEXT_SLICES = 3


def load_model(path):
    """Loads the U-Net model from a state file."""
    try:
        
        model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )
        
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state['model_state'])
        model.to(DEVICE)
        model.eval()
        log.info(f"Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        log.error(f"Model file not found at {MODEL_PATH}. Please ensure the model exists.")
        return None
    except Exception as e:
        log.error(f"Error loading model: {e}")
        return None


segmentation_model = load_model(MODEL_PATH)



def handle_uploaded_ct_files(mhd_file, raw_file):
    """
    Gradio saves uploaded files with temporary names. This function creates a
    temporary directory, copies the files with their correct names based on the
    .mhd header, and returns the path to the new .mhd file.
    """
    if mhd_file is None or raw_file is None:
        return None, None

    temp_dir = tempfile.mkdtemp()
    mhd_path = mhd_file.name
    raw_path = raw_file.name

    
    with open(mhd_path, 'r', errors='ignore') as f:
        mhd_content = f.read()
    
    match = re.search(r'ElementDataFile\s*=\s*(.*)', mhd_content)
    if not match:
        raise ValueError("Could not find 'ElementDataFile' in the .mhd header.")
    
    expected_raw_filename = match.group(1).strip()
    
    
    new_mhd_path = os.path.join(temp_dir, os.path.basename(mhd_path))
    new_raw_path = os.path.join(temp_dir, expected_raw_filename)

    
    shutil.copy(raw_path, new_raw_path)
    shutil.copy(mhd_path, new_mhd_path)

    log.info(f"Processed files in temporary directory: {temp_dir}")
    return new_mhd_path, temp_dir



def perform_segmentation(mhd_file, raw_file):
    """
    Loads a CT scan, runs segmentation on sample slices, and returns overlay images.
    """
    if segmentation_model is None:
        raise gr.Error("Model is not loaded. Please check the model path and restart.")
    if mhd_file is None or raw_file is None:
        raise gr.Error("Please upload both the .mhd and .raw files.")

    temp_mhd_path, temp_dir = None, None
    output_images = []
    
    try:
        temp_mhd_path, temp_dir = handle_uploaded_ct_files(mhd_file, raw_file)
        
        ct_mhd = sitk.ReadImage(temp_mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_mhd) # Shape: (slices, height, width)

        
        slice_indices = np.linspace(0, ct_array.shape[0] - 1, 6, dtype=int)

        for slice_ndx in slice_indices:
            ct_tensor = torch.zeros((CONTEXT_SLICES * 2 + 1, 512, 512))

            
            start_ndx = slice_ndx - CONTEXT_SLICES
            end_ndx = slice_ndx + CONTEXT_SLICES + 1
            for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
                context_ndx = max(0, min(context_ndx, ct_array.shape[0] - 1))
                ct_tensor[i] = torch.from_numpy(ct_array[context_ndx].astype(np.float32))

            ct_tensor.clamp_(-1000, 1000)
            input_tensor = ct_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prediction_g = segmentation_model(input_tensor)
            
            prediction_a = prediction_g.to('cpu').detach().numpy()[0, 0] > 0.5
            ct_slice_a = ct_tensor[CONTEXT_SLICES].numpy()

            image_a = (ct_slice_a + 1000) / 2000
            image_a = np.clip(image_a, 0, 1)
            image_a = np.stack([image_a] * 3, axis=-1) # Convert to RGB
            
            image_a[..., 1] = np.maximum(image_a[..., 1], prediction_a)
            
            output_images.append((image_a, f"Slice {slice_ndx}"))

        return output_images
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            log.info(f"Cleaned up temporary directory: {temp_dir}")



def load_and_store_scan(mhd_file, raw_file):
    """Loads the full CT scan into a NumPy array for the viewer."""
    if mhd_file is None or raw_file is None:
        raise gr.Error("Please upload both the .mhd and .raw files.")
        
    temp_mhd_path, temp_dir = None, None
    try:
        temp_mhd_path, temp_dir = handle_uploaded_ct_files(mhd_file, raw_file)
        ct_mhd = sitk.ReadImage(temp_mhd_path)
        ct_array = sitk.GetArrayFromImage(ct_mhd)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(ct_array[0], cmap='gray', vmin=-200, vmax=200)
        ax.set_title(f"Slice 0 / {ct_array.shape[0]-1}")
        ax.axis('off')
        
        return {
            scan_state: ct_array,
            slice_slider: gr.update(maximum=ct_array.shape[0] - 1, value=0, visible=True),
            slice_plot: fig
        }
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def get_slice(scan_array, slice_index):
    """Displays a specific slice from the loaded CT scan array."""
    if scan_array is None:
        return None
    slice_index = int(slice_index)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(scan_array[slice_index], cmap='gray', vmin=-200, vmax=200)
    ax.set_title(f"Slice {slice_index} / {scan_array.shape[0]-1}")
    ax.axis('off')

    return fig



custom_css = """
body { font-family: 'Helvetica Neue', 'Arial', sans-serif; }
.gradio-container { max-width: 1280px !important; margin: auto !important; }
footer { display: none !important; }
.file-upload-box button {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
    max-height: 150px !important;
    min-height: 50px !important;
    font-size: 12px !important;
}


"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Lung Nodule Segmentation") as demo:
    gr.Markdown("# 🫁 Lung Nodule Segmentation and CT Scan Viewer")

    with gr.Tabs():
        
        with gr.TabItem("Nodule Segmentation Demo"):
            gr.Markdown("Upload a `.mhd` header file and its corresponding `.raw` data file to see the model in action.")
            
            with gr.Row():
                with gr.Column(scale=1):
                   
                    mhd_input_seg = gr.File(label="Upload .mhd File", elem_classes=["file-upload-box"])
                    raw_input_seg = gr.File(label="Upload .raw File", elem_classes=["file-upload-box"])
                    submit_btn_seg = gr.Button("Run Segmentation", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mhd", "sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.raw"],
                            ["sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.mhd", "sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.raw"]
                        ],
                        inputs=[mhd_input_seg, raw_input_seg],
                    )
                
                with gr.Column(scale=2):
                    output_gallery = gr.Gallery(
                        label="Segmentation Results",
                        show_label=True, elem_id="gallery", columns=3, height="auto"
                    )

        
        with gr.TabItem("Interactive CT Scan Viewer"):
            gr.Markdown("Upload a CT scan to view it slice-by-slice using the slider.")
            scan_state = gr.State()
            
            with gr.Row():
                with gr.Column(scale=1):
                    
                    mhd_input_vis = gr.File(label="Upload .mhd File", elem_classes=["file-upload-box"])
                    raw_input_vis = gr.File(label="Upload .raw File", elem_classes=["file-upload-box"])
                    load_btn_vis = gr.Button("Load Scan for Viewing", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mhd", "sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.raw"],
                            ["sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.mhd", "sample_scans\\1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.raw"]
                        ],
                        inputs=[mhd_input_vis, raw_input_vis]
                    )
                
                with gr.Column(scale=2):
                    slice_plot = gr.Plot(label="CT Slice")
                    slice_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Slice", interactive=True, visible=False)

        
        with gr.TabItem("About"):
            gr.Markdown(
                """
                ### Model Architecture
                This application uses a **U-Net** architecture for semantic segmentation of lung nodules.
                - **Encoder-Decoder Structure:** Captures context and enables precise localization.
                - **Skip Connections:** Merge features from the encoder to the decoder to recover fine-grained details.
                - **Input:** 7-channel input (target slice and 3 context slices on each side).
                - **Output:** A single-channel probability mask for nodules.
                
                ### Training Performance
                The model was trained on the LUNA16 dataset.
                """
            )
            with gr.Row():
                gr.Image("assets/scalars_loss.png", label="Training & Validation Loss", elem_classes=["graph"])
                gr.Image("assets/scalars_correct.png", label="Correct", elem_classes=["graph"])
                #gr.Image("assets/distributions.png", label="Prediction Distributions", elem_classes=["graph"])
                #gr.Image("assets/pr.png", label="PR Curve", elem_classes=["graph"])
            with gr.Row():
                gr.Image("assets/distributions.png", label="Prediction Distributions", elem_classes=["graph"])
                #gr.Image("assets/pr.png", label="PR Curve", elem_classes=["graph"])


    
    submit_btn_seg.click(fn=perform_segmentation, inputs=[mhd_input_seg, raw_input_seg], outputs=output_gallery)
    
    load_btn_vis.click(fn=load_and_store_scan, inputs=[mhd_input_vis, raw_input_vis], outputs=[scan_state, slice_slider, slice_plot])
    slice_slider.input(fn=get_slice, inputs=[scan_state, slice_slider], outputs=slice_plot)


if __name__ == "__main__":
    if segmentation_model is None:
        print("\n---")
        print("ERROR: Could not start Gradio app because the model failed to load.")
        print(f"Please check that your model exists at '{MODEL_PATH}' and is a valid PyTorch state file.")
        print("---\n")
    else:
        demo.launch()