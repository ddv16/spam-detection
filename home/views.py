from django.shortcuts import render
from flask import json
from detect import check_spam
# Create your views here.
def home(request):
    if request.method == "POST":
        text = request.POST.get('text')  # <-- this gets the textarea value
        result = check_spam(text)
        
        result['confidence'] = float(result['confidence'])
        print(result)
        return render(request, 'index.html', { 'result': result })
    return render(request,'index.html')