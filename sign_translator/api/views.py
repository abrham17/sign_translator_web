from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .models import VideoTranslation
from .serializers import VideoTranslationSerializer
import torch

# Load your trained model
MODEL_PATH = "slt/signjoey/model.py"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH, map_location=torch.device(device))
model.eval()

def process_video(file_path):

    dummy_features = torch.randn(1, 128, 512)
    with torch.no_grad():
        output = model(dummy_features)
    return output

class VideoTranslationView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        file = request.FILES['file']
        video_instance = VideoTranslation(video=file)
        video_instance.save()

        # Process the video and get translation
        translation = process_video(video_instance.video.path)
        video_instance.translation = str(translation)
        video_instance.save()

        serializer = VideoTranslationSerializer(video_instance)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
