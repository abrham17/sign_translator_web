from django.shortcuts import render
from django.shortcuts import render, redirect
from django.shortcuts import render
from django.core.files.storage import default_storage
import requests
from django.conf import settings


def upload_view(request):
    result = None
    error = None
    temp_image_url = None

    if request.method == 'POST':
        # Get uploaded file
        image_file = request.FILES.get('image')
        
        if image_file:
            try:
                # Save temporarily
                temp_path = default_storage.save(f'tmp/{image_file.name}', image_file)
                
                # Call API app
                api_url = f'http://localhost:8000/api/process-image/'
                files = {'image': open(default_storage.path(temp_path), 'rb')}
                
                response = requests.post(api_url, files=files)
                
                if response.status_code == 200:
                    result = response.json().get('result')
                    temp_image_url = default_storage.url(temp_path)
                else:
                    error = "API processing failed"
                
                # Clean up temp file
                default_storage.delete(temp_path)
                
            except Exception as e:
                error = f"Error processing image: {str(e)}"

    return render(request, 'home.html', {
        'result': result,
        'error': error,
        'temp_image_url': temp_image_url
    })