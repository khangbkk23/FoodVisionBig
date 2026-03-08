import torch
import torch.nn.functional as F
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .model_services.app import WebappConfig

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

def predict_core(image_file):
    try:
        img = Image.open(image_file).convert("RGB")
        img_tensor = WebappConfig.transforms(img).unsqueeze(0).to(WebappConfig.device)
        
        with torch.no_grad():
            outputs = WebappConfig.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        class_name = WebappConfig.class_names[predicted_idx.item()]
        conf_score = confidence.item() * 100
        return class_name, conf_score
    except Exception as e:
        print(f"[ERR] Inference Logic Failed: {e}")
        return None, 0.0
    

# Web views
def home_view(request):
    context = {'prediction': None, 'confidence': None, 'image_url': None, 'error': None}
    
    if request.method == 'POST':
        if 'food_image' not in request.FILES:
            context['error'] = "Please upload an image file you want to classify."
            return render(request, 'home.html', context)
            
        uploaded_file = request.FILES['food_image']
        
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        context['image_url'] = fs.url(filename)
        class_name, conf_score = predict_core(uploaded_file)
        
        if class_name:
            context['prediction'] = class_name.replace("_", " ").title()
            context['confidence'] = f"{conf_score:.2f}%"
        else:
            context['error'] = "There was an error processing the image. Please try again with a different image."
            
    return render(request, 'home.html', context)

def introduce_view(request):
    return render(request, 'introduce.html')

def contact_view(request):
    return render(request, 'contact.html')

# JWT authentication
class PredictAPIView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.data.get('image')
        if not file_obj:
            return Response({"error": "Cannot find 'image' field in form-data"}, status=400)
            
        class_name, conf_score = predict_core(file_obj)
        
        if class_name:
            return Response({
                "status": "success",
                "prediction": class_name,
                "confidence_score": round(conf_score, 4)
            }, status=200)
        return Response({"error": "Inference Error."}, status=500)