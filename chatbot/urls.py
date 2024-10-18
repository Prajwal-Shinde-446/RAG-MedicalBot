from django.urls import path
from . import views

urlpatterns = [
    path("api/upload_document/", views.upload_doc, name='upload_document'),
    path("api/retrieve_response/", views.chatbot_response, name='chatbot_response'),


]