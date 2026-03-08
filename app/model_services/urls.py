from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from . import views 

urlpatterns = [
    path('', views.home_view, name='home'),
    path('introduce/', views.introduce_view, name='introduce'),
    path('contact/', views.contact_view, name='contact'),

    path('api/v1/predict/', views.PredictAPIView.as_view(), name='api-predict'),

    path('api/auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]