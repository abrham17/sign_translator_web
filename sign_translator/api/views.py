from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

@csrf_exempt  # Disable CSRF for API endpoint
def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get uploaded image
            image_file = request.FILES['image']
            
            # Process image
            image = Image.open(image_file)
            
            # Add your translation logic here
            # Example:
            # result = your_model.predict(image)
            
            # Dummy result
            result = "Sample translation result"
            
            return JsonResponse({
                'status': 'success',
                'result': result
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    }, status=400)