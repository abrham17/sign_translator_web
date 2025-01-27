from django.db import models

class VideoTranslation(models.Model):
    video = models.FileField(upload_to='uploads/')
    translation = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
