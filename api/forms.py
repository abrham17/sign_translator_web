from django import forms
from django.core.validators import FileExtensionValidator

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload Image',
        widget=forms.ClearableFileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        }),
        validators=[
            FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png']),
        ]
    )

    def clean_image(self):
        image = self.cleaned_data.get('image')
        return image