from django import forms
import re
# creating a form
class Add(forms.Form):
    newusername=forms.CharField(widget=forms.TextInput(attrs={
        'class': 'user','id':'newusername'
    }))
    newusereid=forms.CharField(widget=forms.TextInput(attrs={
        'class': 'id','id':'newusereid'
    }))

 