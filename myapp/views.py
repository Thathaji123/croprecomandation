# views.py

from django.shortcuts import render
import joblib
from django.conf import settings
import numpy as np

# Load the model using the path from settings
model = joblib.load(settings.MODEL_PATH)


def home(request):
    return render(request,'home.html')

def predict_crop(request):
    if request.method == 'POST':
        # Extract data from the form
        nitrogen = float(request.POST.get('Nitrogen'))
        phosphorus = float(request.POST.get('Phosphorus'))
        potassium = float(request.POST.get('Potassium'))
        temperature = float(request.POST.get('Temperature'))
        humidity = float(request.POST.get('Humidity'))
        ph = float(request.POST.get('pH'))
        rainfall = float(request.POST.get('Rainfall'))

        # Prepare the input for the model
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Make a prediction
        prediction = model.predict(input_data)

        # Pass the result to the template
        return render(request, 'result.html', {'result': prediction[0]})

    return render(request, 'index.html')

