from django.http.response import HttpResponse
from django.urls import path
from . import views
from django.shortcuts import render



def home(request):
    return render(request, 'home.html')

def result(request):
    return render(request, 'result.html')