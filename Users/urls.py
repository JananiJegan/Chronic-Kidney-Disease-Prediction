from django.urls import path
from .import views

urlpatterns=[
    path('',views.index,name='homepage'),
    path('register',views.register,name="kathir"),
    path('login',views.login,name="login"),
    path('logout',views.logout,name="logout"),
    path('data',views.data,name='datapage'),
    path('predict', views.predict, name='predictpage'),
    path('recommend', views.recommend_hospital, name='recommend_hospital'),
]