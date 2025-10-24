import os
import re
import glob
import zlib
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from astropy.time import Time
from sunpy.coordinates.sun import B0, P
from torchvision.transforms import functional as F
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import base64
from io import BytesIO
from flask_cors import CORS
from torch.utils.tensorboard.summary import image
from torchvision import transforms
import matplotlib.patches as patches
import cv2
from flask import Flask, request, jsonify, Response

# Constants
R_SUN_Mm    = 696                    # Solar radius n Mm
min_area_px = 50                     # Minimum pixel area to consider
shrink_factor_suvi = 0.93            # Shrink factor for SUVI disk
image_size = (512, 512)              # image size for resizing

# Morphological kernels
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

##############################################
# Dataset segmentation parameters
##############################################

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

        assert len(self.image_paths) == len(self.mask_paths), (
            f"Number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) do not match."
       )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image = transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

##############################################
# Detect disk
##############################################

def detect_aia_disk_hough(img: np.ndarray):
    """HoughCircles for AIA: returns (cx, cy, r)."""
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(img.shape) != 2:
        raise ValueError("Input image must be single-channel (grayscale).")
    blur = cv2.medianBlur(img, 5)
    circs = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=img.shape[0]//2,
        param1=100, param2=30,
        minRadius=int(img.shape[1]*0.35), maxRadius=int(img.shape[1]*0.6)
    )
    if circs is None:
        return None
    x, y, r = circs[0][0]
    return (x, y, r)


##############################################
# Fit
##############################################

def fit_circle_to_limb(mask: np.ndarray):
    """Least-squares circle fit to mask contour: returns (cx, cy, r)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).squeeze().astype(np.float64)
    x, y = cnt[:,0], cnt[:,1]
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    Suu, Suv, Svv = (u*u).sum(), (u*v).sum(), (v*v).sum()
    Suuu, Svvv = (u**3).sum(), (v**3).sum()
    Suvv, Svuu = (u*v*v).sum(), (v*u*u).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([(Suuu + Suvv)/2.0, (Svvv + Svuu)/2.0])
    uc, vc = np.linalg.solve(A, B)
    cx, cy = xm + uc, ym + vc
    r = np.sqrt(uc*uc + vc*vc + (Suu+Svv)/cnt.shape[0])
    return (cx, cy, r)

##############################################
# Detect SU
##############################################

def detect_suvi_disk_precise(img: np.ndarray):
    """Otsu + LS fit for SUVI: returns (cx, cy, r)."""
    blur = cv2.GaussianBlur(img, (9,9), 0)

    img_8bit = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == best).astype(np.uint8) * 255
    return fit_circle_to_limb(mask)

##############################################
# Calculate
##############################################

def calculate_limb_totals(paths, detector_fn, shrink_factor=1.0, resize_shape = image_size):
    """
    Returns a list of disk‐areas (px²) for each image in `paths`:
      - read & resize
      - detect (cx,cy,r)
      - apply shrink_factor to r
      - compute area = π * r^2
    Non‐detected images yield np.nan.
    """
    totals = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, resize_shape)
        res = detector_fn(img)
        if res is None:
            totals.append(np.nan)
            continue
        _, _, r = res
        totals.append(np.pi * (r * shrink_factor)**2)
    return totals


##############################################
# Calc stats
##############################################

def calculate_limb_stats(paths, detector_fn, shrink_factor=1.0, resize_shape=image_size):
    """
    Runs calculate_limb_totals, then returns (mean_area, std_area)
    ignoring any np.nan entries.
    """
    totals = np.array(
        calculate_limb_totals(paths, detector_fn, shrink_factor, resize_shape)
    )
    # filter out failed detections
    valid = totals[~np.isnan(totals)]
    mean_area = float(valid.mean())
    std_area  = float(valid.std(ddof=1))
    return mean_area, std_area

##############################################
# Save tensors as images
##############################################

def save_tensor_as_image(tensor, file_path):
    image = tensor.squeeze().cpu().numpy()  # Convert tensor to numpy array
    image = (image * 255).astype('uint8')  # Scale to 0-255
    Image.fromarray(image).save(file_path)


##############################################
# Visualize predictions
##############################################

def visualize_predictions(generator, pred_bin, image, threshold = 0.07):

    generator.eval()

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # 3 columns

    inp_img = image.squeeze().cpu().numpy()

    vmin = inp_img.min()
    vmax = inp_img.max()

    axs[0].imshow(inp_img, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Input Image')
    axs[0].axis('off')

    pred_mask = pred_bin.squeeze().cpu().numpy()

    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title('Pix2Pix Prediction')
    axs[1].axis('off')

    overlay = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    overlay[..., 0] = pred_mask
    axs[2].imshow(inp_img, cmap='gray', vmin=vmin, vmax=vmax)
    axs[2].imshow(overlay, alpha=0.5)
    axs[2].set_title('Overlay')
    axs[2].axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    # Encode the image in Base64
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Close the figure to free memory
    plt.close(fig)

    return encoded_image

##############################################
# Evaluate model
##############################################

def evaluate_model(generator, dataloader, device, threshold=0.07):

    generator.eval()

    with torch.no_grad():
        for inp in dataloader:
            inp = inp.to(device)

            pred = generator(inp)
            pred_bin = (pred > threshold).float()

            pred_np = pred_bin.cpu().numpy()

    return pred_np

##############################################
# Analyze coronal holes
##############################################

def analyze_structures(image, pred_bin, instrument: str, date, time) -> dict:
    """
    Analyze a single coronal structure and return a JSON-serializable dict with:
      - filename, observation time, B0, P
      - list of holes, each with label, area_mm2, centroid_px, offset_Mm, latitude, longitude
    """

    dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
    obs_time = dt.isoformat() + 'Z'

    # compute solar tilt parameters
    t = Time(dt.isoformat(), scale='utc')
    B0_deg = B0(t).value
    P_deg  = P(t).value
    B0_rad = np.radians(B0_deg) #nove
    P_rad = np.radians(P_deg) #nove

    # load image and mask
    img = image.squeeze().cpu().numpy()

    #image_bytes = image.squeeze().cpu().numpy()
    #nparr = np.frombuffer(image_bytes, np.uint8)
    #img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE) # updated grayscaling

    mask = pred_bin.squeeze().cpu().numpy()

    if img is None or mask is None:
        raise FileNotFoundError(f"Missing file: {image} or {pred_bin}")
    mask_bin = (mask > 0).astype(np.uint8)

    # detect solar disk
    #if instrument.lower() == 'aia':
    #    cx, cy, r_px = detect_aia_disk_hough(img)
    #else:
    #    cx, cy, r_px = detect_suvi_disk_precise(img)
    #    r_px *= shrink_factor_suvi

    # new detection of solar disk

    inst_lower = instrument.lower()
    if inst_lower == 'aia':
        res = detect_aia_disk_hough(img)
    else:
        res = detect_suvi_disk_precise(img)

    if res is None:
        raise ValueError("Solar disk not detected.")

    cx, cy, r_px = res
    if inst_lower != 'aia':
        # Kalibračné zmenšenie SUVI polomeru (ovplyvní scale/metry)
        r_px *= shrink_factor_suvi

    if r_px <= 0:
        raise ValueError(f"Detected non-positive radius r_px={r_px}")

    # pixels -> Mm
    scale = R_SUN_Mm / r_px  # Mm per pixel

    # morphological cleaning
    m_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_kernel)
    m_opened = cv2.morphologyEx(m_closed, cv2.MORPH_OPEN, open_kernel) #added
    m_proc = (m_opened > 0).astype(np.uint8)
    # m_proc   = cv2.morphologyEx(m_closed, cv2.MORPH_OPEN,  open_kernel)

    # label and filter small
    labeled = label(m_proc, connectivity=2)

    holes = []
    for prop in regionprops(labeled):
        if prop.area < min_area_px:
            continue

        # compute metrics
        # area_mm2 = prop.area * scale**2
        # cy_px, cx_px = prop.centroid
        # x_mm = (cx_px - cx) * scale
        # y_mm = (cy   - cy_px) * scale
        # correct for tilt and compute heliographic coords
        # B0_rad = np.radians(B0_deg)
        # P_rad  = np.radians(P_deg)
        # xp =  x_mm * np.cos(P_rad) + y_mm * np.sin(P_rad)
        # yp = -x_mm * np.sin(P_rad) + y_mm * np.cos(P_rad)
        # lat_rad = np.arcsin(yp / R_SUN_Mm) + B0_rad
        # cos_lat = np.cos(lat_rad)
        # lon_rad = np.arcsin(xp / (R_SUN_Mm * cos_lat))
        # lat_deg = np.degrees(lat_rad)
        # lon_deg = np.degrees(lon_rad)

        # regionprops centroid: (row=y, col=x)
        cy_px_label, cx_px_label = prop.centroid
        x_mm = (cx_px_label - cx) * scale  # +x doprava
        y_mm = (cy - cy_px_label) * scale  # +y nahor (invert Y osi obrazu)

        # rotácia o P (slnečný sever hore)
        xp = x_mm * np.cos(P_rad) + y_mm * np.sin(P_rad)
        yp = -x_mm * np.sin(P_rad) + y_mm * np.cos(P_rad)

        # normalizované súradnice (v polomeroch Slnka)
        x1 = xp / R_SUN_Mm
        y1 = yp / R_SUN_Mm

        # vzdialenosť od stredu na disku; ochrana proti numerike
        rho2 = np.clip(x1 * x1 + y1 * y1, 0.0, 1.0)
        z = np.sqrt(1.0 - rho2)  # "hĺbka" (cos uhlovej vzdialenosti)

        # Stonyhurst latitude
        lat_rad = np.arcsin(z * np.sin(B0_rad) + y1 * np.cos(B0_rad))

        # CMD (central meridian distance), západ kladný; plný rozsah cez atan2
        num = x1
        den = z * np.cos(B0_rad) - y1 * np.sin(B0_rad)
        lon_cmd_rad = np.arctan2(num, den)

        lat_deg = float(np.degrees(lat_rad))
        lon_deg = float(np.degrees(lon_cmd_rad))


        holes.append({
            'label':       int(prop.label),
            'area_mm2':    float(prop.area * scale**2),
            'centroid_px': [float(cx_px_label), float(cy_px_label)],
            'offset_Mm':   [float(x_mm), float(y_mm)],
            'latitude':    float(lat_deg),
            'longitude':   float(lon_deg)
        })

    result = {
        #'filename':     fname,
        'obs_time':     obs_time,
        'instrument':   instrument,
        'B0_deg':       float(B0_deg),
        'P_deg':        float(P_deg),
        'disk': {'cx': float(cx), 'cy': float(cy), 'r_px': float(r_px)}, #added
        'holes':        holes
    }
    return result

##############################################
# Plot structure
##############################################

def plot_structures(generator, mask, image, instrument: str, data: dict):
    """
    Plot the original image, mask of the prediction, and image with holes using the JSON result.
    """
    # Convert tensors to numpy arrays

    generator.eval()

    img = image.squeeze().cpu().numpy()
    #image_bytes = image.squeeze().cpu().numpy()
    #nparr = np.frombuffer(image_bytes, np.uint8)
    #img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    mask_bin = mask.squeeze().cpu().numpy()

    # Detect disk for plotting
    #if instrument.lower() == 'aia':
    #    cx, cy, r_px = detect_aia_disk_hough(img)
    #else:
    #    cx, cy, r_px = detect_suvi_disk_precise(img)
    #    r_px *= shrink_factor_suvi

    # použi disk z analýzy (fallback na detekciu, ak by chýbal)
    cx = data.get('disk', {}).get('cx')
    cy = data.get('disk', {}).get('cy')
    r_px = data.get('disk', {}).get('r_px')
    if cx is None or cy is None or r_px is None:
        inst_lower = instrument.lower()
        if inst_lower == 'aia':
            res = detect_aia_disk_hough(img)
        else:
            res = detect_suvi_disk_precise(img)
        if res is None:
            raise ValueError("Solar disk not detected (plot).")
        cx, cy, r_px = res
        if inst_lower != 'aia':
            r_px *= shrink_factor_suvi

    # Prepare the figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # 3 columns

    # Plot the original image
    vmin = img.min()
    vmax = img.max()

    axs[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot the mask of the prediction
    axs[1].imshow(mask_bin, cmap='gray')
    axs[1].set_title('Prediction Mask')
    axs[1].axis('off')



    # Plot the image with holes
    #overlay = np.zeros((*mask_bin.shape, 3), dtype=np.float32)
    #overlay[..., 0] = mask_bin  # Red channel for holes
    #axs[2].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    #axs[2].imshow(overlay, alpha=0.5)

    axs[2].imshow(img, cmap='gray')
    circ = patches.Circle((cx, cy), r_px, edgecolor='white', facecolor='none', linewidth=2)
    axs[2].add_patch(circ)

    # obrysy dier (rovnaká morfológia a binarita ako v analýze)
    m_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_kernel)
    m_opened = cv2.morphologyEx(m_closed,  cv2.MORPH_OPEN,  open_kernel)
    m_proc   = (m_opened > 0).astype(np.uint8)
    axs[2].contour(m_proc, levels=[0.5], colors='red', linewidths=1.5)

    axs[2].set_title('Image with detected structures')
    axs[2].axis('off')

    # Annotate holes
    #for hole in data['holes']:
    for hole in data.get('holes', []):
        cx_px, cy_px = hole['centroid_px']
        lat, lon = hole['latitude'], hole['longitude']
        area_mm2 = hole['area_mm2']
        #axs[2].plot(cx_px, cy_px, 'o', markeredgecolor='yellow', markerfacecolor='none', markersize=10)
        #axs[2].text(cx_px + 5, cy_px + 5,
        #            f"{hole['label']}: {area_mm2:.1f}Mm²",
        #            color='yellow', fontsize=7)

        axs[2].text(cx_px+5, cy_px+5,
                f"{hole['label']}: {area_mm2:.1f}Mm²\n{lat:.1f}°, {lon:.1f}°",
                color='yellow', fontsize=9, weight='bold')
    # Add title
    title = f"{data['obs_time']}  B0={data['B0_deg']:.2f}° P={data['P_deg']:.2f}°"

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    #plt.show()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    # Encode the image in Base64
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Close the figure to free memory
    plt.close(fig)

    return encoded_image


##############################################
# Extract key-value from JSON
##############################################

def extract_key_value(json_data, key):
    value = json_data.get(key)
    return value

##############################################
# APP
##############################################

app = Flask("SCSSNet")
CORS(app)

@app.route('/predict', methods=['POST'])

def predict():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    data = request.get_json()

    IMG_WIDTH, IMG_HEIGHT = 1024, 1024

    tasktype = extract_key_value(data, "tasktype")
    threshold_param = extract_key_value(data, "threshold")
    test_image = extract_key_value(data, "image")
    instrument = extract_key_value(data, "instrument")
    date = extract_key_value(data, "date")
    time = extract_key_value(data, "time")

    image = Image.open(BytesIO(base64.b64decode(test_image))).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input = transform(image)

    # For Active Region (AR) detection, we use a specific model.
    if tasktype == "AR":

        if threshold_param == "conservative":
            threshold = 0.01
        if threshold_param == "medium":
            threshold = 0.4
        if threshold_param == "nonconservative":
            threshold = 0.9

        generator = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=1,
            classes=1,
            activation='sigmoid',
            decoder_attention_type='scse',
        ).to(device)

        model_path = 'models/checkpoint_epoch_51.pth.tar'
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

        pred = generator(input.unsqueeze(0))
        pred_bin = (pred > threshold).float()

        pred_np = pred_bin.cpu().numpy()

    #+
    else:

        if threshold_param == "conservative":
            threshold = 0.0001
        if threshold_param == "medium":
            threshold = 0.5
        if threshold_param == "nonconservative":
            threshold = 0.99

        generator = smp.Unet(
            encoder_name='resnet34', encoder_weights='imagenet',
            in_channels=1, classes=1, activation=None,
            decoder_attention_type='scse'
        ).to(device)

        # Load the pre-trained model

        model_path = 'models/GAN_epoch39_IoU0.7187.pth'
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint['G'])
        generator.eval()

        pred = generator(input.unsqueeze(0))
        pred_bin = (pred > threshold).float()

        pred_np = pred_bin.cpu().numpy()

    print(pred_np)

    temp_image_path = "temp_image.png"
    save_tensor_as_image(input, temp_image_path)

    stats = analyze_structures(input, pred_bin, instrument, date, time)

    encoded_prediction = plot_structures(generator, pred_bin, input, instrument, stats)

    mask_array = pred_bin.squeeze().cpu().numpy().astype('uint8')
    compressed_mask = zlib.compress(mask_array.tobytes())
    encoded_mask = base64.b64encode(compressed_mask).decode('utf-8')

    print(encoded_mask)

    output_formatted = {
        'image': encoded_prediction,
        'mask': encoded_mask,
        'threshold': threshold,
        'tasktype': tasktype,
        'date': date,
        'time': time,
        'stats': stats
    }

    response = Response(
        response= json.dumps(output_formatted),
        status=200,
        mimetype='application/json'
    )
    print(response)
    return response
    os.remove(temp_image_path)

if __name__ == '__main__':
    app = app
    app.run()
