from flask import Flask, request, jsonify
from astropy.io import fits
from PIL import Image
import sunpy.map
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
from flask_cors import CORS


def fits_to_png(data, header):
    """
    Convert FITS data and header to a PNG image (base64-encoded) using SunPy's plotting and percentile normalization.
    """
    solar_map = sunpy.map.Map((data, header))
    vmin = np.percentile(data, 44.35)
    vmax = np.percentile(data, 99.99)
    solar_map.plot_settings['norm'].vmin = vmin
    solar_map.plot_settings['norm'].vmax = vmax

    fig = plt.figure(figsize=(10.24, 10.24))
    ax = plt.axes([0, 0, 1, 1], projection=solar_map)
    solar_map.plot()
    ax.grid(False)
    plt.axis("off")
    plt.title(label="")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    png_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return png_base64


app = Flask(__name__)
CORS(app)

@app.route('/convert-fits', methods=['POST'])
def convert_fits():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    fits_file = request.files['file']

    try:
        file_bytes = fits_file.read()
        file_buffer = io.BytesIO(file_bytes)

        with fits.open(file_buffer) as hdul:

            hdu_with_data = next((hdu for hdu in hdul if hdu.data is not None), None)
            if hdu_with_data is None:
                return jsonify({'error': 'No image data found in FITS file.'}), 400

            data = hdu_with_data.data
            header = hdu_with_data.header

            png_base64 = fits_to_png(data, header)

            instrument = header.get('INSTRUME', 'Unknown')
            if instrument and 'goes-r series solar ultraviolet imager' in instrument.lower():
                instrument = 'suvi'

            date_obs = header.get('DATE-OBS', 'Unknown')
            if date_obs != 'Unknown' and 'T' in date_obs:
                date, time = date_obs.split('T', 1)
            else:
                date = date_obs
                time = header.get('TIME-OBS', 'Unknown')

            metadata = {
                'instrument': instrument,
                'date': date,
                'time': time
            }

        return jsonify({'png': png_base64, 'metadata': metadata})

    except Exception as e:
        print("Error processing FITS:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    app = app
    app.run(host='0.0.0.0', port=5500)