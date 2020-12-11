from django.http.response import HttpResponse
from django.shortcuts import render
import joblib



def home(request):
    return render(request, 'home.html')

def result(request):
    classifier = joblib.load('final_model.sav')
    
    lis = []

    lis.append(request.GET['variance'])
    lis.append(request.GET['skewness'])
    lis.append(request.GET['curtosis'])
    lis.append(request.GET['entropy'])

    ans = classifier.predict([lis])


    return render(request, 'result.html', {'ans':ans})