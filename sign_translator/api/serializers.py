from rest_framework import serializers
from .models import VideoTranslation , ImageTranslation, WebcamTranslation

class VideoTranslationSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoTranslation  # The model this serializer is for
        fields = ['id', 'video', 'translation', 'uploaded_at']  # The fields to include in serialization
class ImageTranslationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageTranslation
        fields = ['id', 'image', 'translation', 'uploaded_at']
class WebcamTranslationSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebcamTranslation
        fields = ['id', 'translation', 'uploaded_at']