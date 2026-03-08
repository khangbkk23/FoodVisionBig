from django.urls import path
from . import views

urlpatterns = [
	path('', views.home_view,name='home'),
 	path('predict/', views.predict_view, name='predict'),
	path('contact/', views.contact_view, name='contact'),
	path('about/', views.about_view, name='about'),
	path('app/', views.app_view, name='app'),
	path('api/login/', views.login_view, name='login_api'),
]
