import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
import os

class Preprocessor:

    '''

        Args -> file.nii 3D
        Restituisce un nuovo file .nii "preprocessed", sempre 3D sia per le maschere che per le immagini. 
        Le immagini iniziali subiranno: Resize, Outlier Remouval e normalizzazione.
        Salvo i file.nii se necessario.

    '''

    def __init__(self, filename):
        self.filename = filename
        self.img = nib.load(filename)
        self.data = self.img.get_fdata()  # (H, W, D) o (H, W, D, T) se 4D
        self.affine = self.img.affine  

    def visualize_slices(self, num_slices=5):
        """Visualizza alcune slice del volume."""
        slice_indices = np.linspace(0, self.data.shape[2] - 1, num_slices, dtype=int)

        fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
        for i, idx in enumerate(slice_indices):
            axes[i].imshow(self.data[:, :, idx], cmap="gray")
            axes[i].set_title(f"Slice {idx}")
            axes[i].axis("off")
        plt.show()

    def resize(self, size=160, is_mask=False):
        """Ridimensiona le slice a 160x160 mantenendo tutte le slice."""
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized_slices = [
            cv2.resize(self.data[:, :, i], (size, size), interpolation=interpolation)
            for i in range(self.data.shape[2])  # Nessuna slice viene esclusa
        ]
        return np.stack(resized_slices, axis=-1)  # Ricostruisce il volume


    def remove_outliers(self, data, percentile=80):
        """Sostituisce i pixel fuori dal range 0-70% con la soglia inferiore o superiore."""
        threshold = np.percentile(data, percentile)
        data[data > threshold] = threshold  # Setta i pixel sopra soglia a threshold
        return data

    def normalize(self, data):
        """Normalizza l'immagine tra 0 e 1."""
        data = data - np.min(data)  # Shift valori minimi a 0
        data = data / np.max(data) if np.max(data) != 0 else data  # Normalizzazione [0,1]
        return data
    
   

    def resize_volume(self, new_shape=(160, 160, None)):
        # Ridimensiona l'intero volume preservando la qualità.
        original_shape = self.data.shape
        scale_factors = [new_shape[i] / original_shape[i] if new_shape[i] else 1 for i in range(3)]
        resized_data = zoom(self.data, scale_factors, order=3)  # Interpolazione cubica
        return resized_data

    
    def preprocess(self):
        """Applica tutte le operazioni di preprocessing in sequenza e salva un nuovo file .nii"""
        resized_data = self.resize(is_mask=True)
        outlier_removed = self.remove_outliers(resized_data)
        normalized_data = self.normalize(outlier_removed)
        new_filename = self.filename.replace(".nii", "_preprocessed.nii")
        new_img = nib.Nifti1Image(normalized_data, affine=self.affine)
        nib.save(new_img, new_filename)
        print(f"Preprocessing completato! Dati salvati in: {new_filename}")

 
    def visualize_multiple_slices_pre_and_post(self, num_slices=5):
        """Visualizza più slice prima e dopo il preprocessing."""
        processed_data = self.normalize(self.remove_outliers(self.resize()))  # Pipeline

        slice_indices = np.linspace(0, self.data.shape[2] - 1, num_slices, dtype=int)  # Ora include prima/ultima slice

        fig, axes = plt.subplots(2, num_slices, figsize=(15, 6))  # 2 righe: Originale | Preprocessata
        for i, idx in enumerate(slice_indices):
            # Slice originale
            axes[0, i].imshow(self.data[:, :, idx], cmap="gray")
            axes[0, i].set_title(f"Originale - Slice {idx}")
            axes[0, i].axis("off")

            # Slice preprocessata (stesso indice per allineamento corretto)
            axes[1, i].imshow(processed_data[:, :, idx], cmap="gray")
            axes[1, i].set_title(f"Preprocessata - Slice {idx}")
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.show()
    
    def visualize_overlay(self, num_slices=5):
        """Visualizza sovrapposte le maschere e le immagini iniziali."""
        # Assicurati che le maschere siano preprocessate come le immagini
        mask_filename = self.filename.replace(".nii", "_preprocessed_mask.nii")
        mask_img = nib.load(mask_filename)
        mask_data = mask_img.get_fdata()

        # Seleziona alcuni slice per la visualizzazione
        slice_indices = np.linspace(0, self.data.shape[2] - 1, num_slices, dtype=int)

        fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
        for i, idx in enumerate(slice_indices):
            ax = axes[i]
            
            # Visualizza l'immagine originale
            ax.imshow(self.data[:, :, idx], cmap="gray", alpha=0.7)  # L'immagine ha un'alpha per essere visibile sotto
            
            # Sovrappone la maschera (con trasparenza) sopra l'immagine
            ax.imshow(mask_data[:, :, idx], cmap="jet", alpha=0.3)  # La maschera con un'alpha più bassa per la trasparenza
            
            ax.set_title(f"Overlay - Slice {idx}")
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()


    """
        Questa parte di codice si occuperà di preprocessare le maschere semantiche.
        Le immagini delle maschere sono già in scala 0-1, non ci sarà bisogno di eliminare outliers o normalizzare. Basterà farne il resize.
    """
        
    def preprocess_gt(self):
        """Applica il solo ridimensionamento alle maschere"""
        resized_data = self.resize(is_mask=True)  
        new_img = nib.Nifti1Image(resized_data, affine=self.affine)
        new_filename = self.filename.replace(".nii", "_preprocessed.nii")
        nib.save(new_img, new_filename)
        print(f"Preprocessing maschera completato! Salvato in: {new_filename}")

    def visualize_multiple_slices_pre_and_post_gt(self, num_slices=5):
        """Visualizza più slice prima e dopo il preprocessing."""
        processed_data = self.resize()  # Pipeline
        slice_indices = np.linspace(0, self.data.shape[2] - 1, num_slices, dtype=int)  # Ora include prima/ultima slice
        fig, axes = plt.subplots(2, num_slices, figsize=(15, 6))  # 2 righe: Originale | Preprocessata
        for i, idx in enumerate(slice_indices):
            # Slice originale
            axes[0, i].imshow(self.data[:, :, idx], cmap="gray")
            axes[0, i].set_title(f"Originale - Slice {idx}")
            axes[0, i].axis("off")

            # Slice preprocessata (stesso indice per allineamento corretto)
            axes[1, i].imshow(processed_data[:, :, idx], cmap="gray")
            axes[1, i].set_title(f"Preprocessata - Slice {idx}")
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.show()


def visualize_overlay_with_gt(image_path, gt_path, num_slices=5):
    """Visualizza le immagini e le maschere sovrapposte per 5 slice."""
    
    # Carica l'immagine e la maschera
    img = nib.load(image_path)
    img_data = img.get_fdata()
    
    gt = nib.load(gt_path)
    gt_data = gt.get_fdata()

    # Seleziona gli indici delle slice da visualizzare
    slice_indices = np.linspace(0, img_data.shape[2] - 1, num_slices, dtype=int)

    # Creiamo la figura per la visualizzazione
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    
    for i, idx in enumerate(slice_indices):
        # Immagine
        axes[i].imshow(img_data[:, :, idx], cmap="gray", alpha=0.7)  # Immagine di base in grigio con trasparenza
        # Sovrapponi la maschera (GT) con colori diversi
        axes[i].imshow(gt_data[:, :, idx], cmap="jet", alpha=0.3)  # Maschera in jet con trasparenza
        axes[i].axis("off")  # Rimuovi gli assi
        axes[i].set_title(f"Slice {idx}")

    plt.tight_layout()
    plt.show()

# Esempio di utilizzo:


'''
# Carica una maschera grezza
file_nifti2 = "database/training/patient001/patient001_frame01_gt_preprocessed.nii.gz"
mask = nib.load(file_nifti2).get_fdata()

# Valori unici presenti
unique_values = np.unique(mask)
print("Valori unici nella maschera preprocessata:", unique_values)
'''


# **Questo serve per eliminare eventuali Rimmate**
'''
path = "/Users/emanueleamato/Downloads/Cardiac_MRI/database/"

for root, dirs, files in os.walk(path):
    for file in files:
        if "_preprocessed.nii" in file:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Eliminato: {file_path}")

print("Eliminazione completata.")
'''

'''
Test di funzioni 

file_nifti = "database/training/patient001/patient001_frame01.nii.gz"
preproc = Preprocessor(file_nifti)

preproc.preprocess()  # Salva il nuovo file pre-elaborato
preproc.visualize_multiple_slices_pre_and_post()  # Confronto prima/dopo

file_nifti2 = "database/training/patient001/patient001_frame01_gt.nii.gz"
preproc2 = Preprocessor(file_nifti2)
preproc2.preprocess_gt()  # Salva il nuovo file pre-elaborato
preproc2.visualize_multiple_slices_pre_and_post_gt()  # Confronto prima/dopo

image_path = "database/training/patient001/patient001_frame01_preprocessed.nii.gz"
gt_path = "database/training/patient001/patient001_frame01_gt_preprocessed.nii.gz"
visualize_overlay_with_gt(image_path, gt_path)
# ** Questo serve per controllare che il preprocessing sia fatto bene 

'''

# Preprocesso tutti i file nel training e nel test (cambia i path).



path="/Users/emanueleamato/Downloads/Cardiac_MRI/database/testing"

for root,dirs,files in os.walk(path):
    for file in files:
        if (".nii") in file and ("_4d") not in file and ("_gt"):
            new_path=os.path.join(root,file)
            print(f'Processando il file {new_path}: ')
            preproc=Preprocessor(new_path)
            preproc.preprocess() 
        if ("_gt.nii") in file:
            new_path=os.path.join(root,file)
            print(f'Processando il file _gt{new_path}: ')
            preproc=Preprocessor(new_path)
            preproc.preprocess_gt() 




            


