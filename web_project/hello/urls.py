
from django.urls import path
from hello import views


urlpatterns=[
    path("",views.home,name='home'),
    path("start",views.start,name='start'),
    path("add",views.add,name='add'),

]