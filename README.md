# Solar Coronal Structure Segmentation (SCSSNet)

A Python project for segmenting and analyzing solar coronal structures using deep learning.  
Includes a Flask API backend and a client for interacting with the service.

## Project Structure

project-root/ │ ├── service/ # Flask API backend 
│ ├── app.py │ ├── models/ │ └── ... │ ├── client/ # API client or frontend │ ├── client.py │ └── ... │ ├── requirements.txt └── README.md


## Features

- Deep learning segmentation of solar images (Active Regions, Coronal Holes)
- REST API for predictions and analysis
- Image preprocessing, disk detection, and region analysis
- Returns annotated images and statistics

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Install client dependencies (if using React/JS):
    ```bash
    cd client
    npm install
   ```
   
## Usage

1. **Start the Flask API server:**
   ```bash
   cd service
    python service/app.py
   ```
The API will be available at http://localhost:5000.

2. **Use the client to interact with the API:**
   - If using a web client, run:
     ```bash
     cd flask-client
     npm start
     ```

### API Endpoints
- POST /predict

Send a JSON payload with image data and parameters to get segmentation results.
Example request:
```json
{
  "tasktype": "AR",
  "threshold": "medium",
  "image": "<base64-encoded PNG>",
  "instrument": "AIA",
  "date": "2024-06-01",
  "time": "12:00:00"
  }
}
```

## Model Files

Place pre-trained model files in service/models/ as required by app.py. Two models are needed, one for Active Regions (AR) and one for Coronal Holes (CH).

