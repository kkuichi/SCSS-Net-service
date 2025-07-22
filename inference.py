import requests
import base64
import matplotlib.pyplot as plt
from io import BytesIO

# URL of the service
url = 'http://127.0.0.1:5000/predict'

test = {}
# load test JSON
with open('/service/test_data/AR/images_renamed/test/suvi_ar_2023_10_14T17_59_45.png', 'rb') as file:
    base64_bytes = base64.b64encode(file.read()).decode('utf-8')


threshold = "medium"
tasktype = "AR"

data = {

    'image': base64_bytes,
    'threshold': threshold,
    'tasktype': tasktype

}

print(data)

# Make a POST request to the prediction endpoint with customer test_data
response = requests.post(url, json=data)

if response.status_code == 200:
    print('JSON test_data sent successfully!')
    response_data = response.json()

    # Decode the Base64 image from the response
    if 'image' in response_data:
        decoded_image = base64.b64decode(response_data['image'])
        image = plt.imread(BytesIO(decoded_image), format='png')

        # Plot the image
        plt.imshow(image)
        plt.axis('off')
        plt.title('Predicted Image')
        plt.show()
    else:
        print('No image found in the response.')
else:
    print('Failed to send JSON test_data:', response.status_code)

print(response.text)