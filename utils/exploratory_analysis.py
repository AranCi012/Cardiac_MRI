import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go

def analyze_nifti_in_a_folder(path):

    """
        Dato un percorso file, mi dice, di tutti i file .nii presenti nella cartella la loro dimensione.
    """
    nii_files = [f for f in os.listdir(path) if f.endswith('.nii') or f.endswith('.nii.gz')]

    for file in nii_files:
        file_path = os.path.join(path, file)
        img = nib.load(file_path)
        data = img.get_fdata()

        print(f"\nFile: {file} - Shape: {data.shape}")

        if len(data.shape) == 3:
            print("🔹 File 3D → Un singolo volume.")
            print(f"   - Valore minimo: {np.min(data)}")
            print(f"   - Valore massimo: {np.max(data)}")
            print(f"   - Media: {np.mean(data)}")

        elif len(data.shape) == 4:
            print("🔹 File 4D → Contiene più frame (es. serie temporale).")
            print(f"   - Numero di frame: {data.shape[3]}")

            # Analizza il primo e l'ultimo frame per vedere le differenze
            print(f"   - Frame 0: Min {np.min(data[:,:,:,0])}, Max {np.max(data[:,:,:,0])}, Media {np.mean(data[:,:,:,0])}")
            print(f"   - Frame {data.shape[3]-1}: Min {np.min(data[:,:,:,-1])}, Max {np.max(data[:,:,:,-1])}, Media {np.mean(data[:,:,:,-1])}")

        else:
            print("❌ Formato non riconosciuto.")

def visualize_single_slice_of_3d_volume_of_nifti_file(filepath, frame_idx=0):

    """Visualizza un singolo frame di un'immagine NIfTI."""
    img = nib.load(filepath)
    data = img.get_fdata()

    print(f"File: {os.path.basename(filepath)} - Shape: {data.shape}")

    if len(data.shape) == 3:
        slice_idx = data.shape[2] // 2  # Slice centrale
        plt.imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
        plt.title(f"Slice centrale di {os.path.basename(filepath)}")
        plt.axis("off")
        plt.show()

    elif len(data.shape) == 4:
        slice_idx = data.shape[2] // 2
        plt.imshow(data[:, :, slice_idx, frame_idx].T, cmap='gray', origin='lower')
        plt.title(f"Frame {frame_idx} - Slice {slice_idx} di {os.path.basename(filepath)}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Formato non supportato: {os.path.basename(filepath)}")

def visualize_nifti_all_frames_in_a_4d_file(filepath):

    """Se il file è 4D, stampa tutti i frame del volume 4D."""
    img = nib.load(filepath)
    data = img.get_fdata()

    if len(data.shape) != 4:
        print(f"Il file {os.path.basename(filepath)} non è 4D.")
        return

    slice_idx = data.shape[2] // 2  # Slice centrale
    num_frames = data.shape[3]

    for frame_idx in range(num_frames):
        plt.imshow(data[:, :, slice_idx, frame_idx].T, cmap='gray', origin='lower')
        plt.title(f"Frame {frame_idx} - Slice {slice_idx} di {os.path.basename(filepath)}")
        plt.axis("off")
        plt.show()

def visualize_nifti_slices(filepath):
    """Visualizza le slice assiali di un file NIfTI."""
    img = nib.load(filepath)
    data = img.get_fdata()

    if len(data.shape) < 3:
        print(f"Formato non supportato: {os.path.basename(filepath)}")
        return

    max_slice = data.shape[2] - 1

    def show_slice(slice_idx):
        plt.imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
        plt.title(f"Slice {slice_idx} di {os.path.basename(filepath)}")
        plt.axis("off")
        plt.show()

    widgets.interact(show_slice, slice_idx=widgets.IntSlider(min=0, max=max_slice, step=1, value=max_slice // 2))


def visualize_nifti_3d(filepath):
    """Visualizza un volume NIfTI in 3D con Plotly."""
    img = nib.load(filepath)
    data = img.get_fdata()

    # Normalizza i valori per la visualizzazione
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    fig = go.Figure(data=go.Volume(
        x=np.linspace(0, data.shape[0], data.shape[0]),
        y=np.linspace(0, data.shape[1], data.shape[1]),
        z=np.linspace(0, data.shape[2], data.shape[2]),
        value=data.flatten(),
        opacity=0.1,  # Trasparenza
        surface_count=15,  # Dettaglio della visualizzazione
    ))

    fig.update_layout(title="Visualizzazione 3D del volume NIfTI")
    fig.show()

    
path="ACDC/database/testing/patient101"
analyze_nifti_in_a_folder(path)