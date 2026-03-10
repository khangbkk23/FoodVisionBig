import torch
import torch.nn.functional as F
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from model_services.apps import ModelServicesConfig

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

def predict_core(image_file):
    try:
        img = Image.open(image_file).convert("RGB")
        img_tensor = ModelServicesConfig.transforms(img).unsqueeze(0).to(ModelServicesConfig.device)
        
        with torch.no_grad():
            outputs = ModelServicesConfig.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
            
        top_predictions = []
        for i in range(5):
            idx = top5_indices[0][i].item()
            conf_score = top5_prob[0][i].item() * 100

            raw_name = ModelServicesConfig.class_names[idx]
            clean_name = raw_name.replace("_", " ").title()
            
            top_predictions.append({
                'name': clean_name,
                'confidence': f"{conf_score:.2f}%"
            })
            
        return top_predictions
    except Exception as e:
        print(f"[ERR] Inference Logic Failed: {e}")
        return None

def home_view(request):
    context = {'top_predictions': None, 'image_url': None, 'error': None}
    
    if request.method == 'POST':
        if 'food_image' not in request.FILES:
            context['error'] = "Vui lòng đính kèm tệp hình ảnh hợp lệ."
            return render(request, 'home.html', context)
            
        uploaded_file = request.FILES['food_image']
        
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        context['image_url'] = fs.url(filename)
        
        top_predictions = predict_core(uploaded_file)
        
        if top_predictions:
            context['top_predictions'] = top_predictions
        else:
            context['error'] = "Đã xảy ra lỗi trong quá trình phân giải ma trận hình ảnh."
            
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