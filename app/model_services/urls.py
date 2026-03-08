from django.urls import path
from . import views

urlpatterns = [
	path('', views.home_view,name='home'),
	path('introduce/', views.introduce_view,name='introduce'),
	path('contact/', views.contact_view,name='contact'),
 
	path('api/v1/predict/', views.PredictAPIView.as_view(), name='api-predict'),
 
	path('api/auth/token/', views.TokenObtainPairAPIView.as_view(), name='token_obtain_pair'),
	path('api/auth/token/refresh/', views.TokenRefreshAPIView.as_view(), name='token_refresh'),
]
