from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage 
from .predict_categories import predict_category

import os
from django.conf import settings

def upload_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_image = UploadedImage(image=request.FILES['image'])
            uploaded_image.save() 
            image_path = uploaded_image.image.path 
            
            top5_categories, top5_probs = predict_category(image_path)

            uploaded_image.delete()
            image_path = os.path.join(settings.MEDIA_ROOT, image_path)
            if os.path.exists(image_path):
                os.remove(image_path)

            return render(request, "results.html", {"predictions": zip(top5_categories, top5_probs)})
    else:
        form = ImageUploadForm()

    return render(request, "upload.html", {"form": form})
        
