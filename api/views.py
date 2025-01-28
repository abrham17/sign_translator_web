from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the image processor and model once at the top-level for efficiency
processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")

@csrf_exempt
def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get the uploaded image file
            image_file = request.FILES['image']
            
            # Open the image using PIL
            with Image.open(image_file) as img:
                # Convert the image to RGB (if it's not already in RGB mode)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize the image to the model's input size (224x224)
                img = img.resize((224, 224))

                # Preprocess the image using the processor
                inputs = processor(images=img, return_tensors="pt")

                # Perform inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get the predicted class index and label
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_class_idx]
                print(predicted_label)
                # Return the prediction as JSON
                return JsonResponse({
                    'status': 'success',
                    'result': predicted_label
                })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)
