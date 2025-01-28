from rest_framework import serializers
from .models import VideoTranslation

class VideoTranslationSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoTranslation  # The model this serializer is for
        fields = ['id', 'video', 'translation', 'uploaded_at']  # The fields to include in serialization
