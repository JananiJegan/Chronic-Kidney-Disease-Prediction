from django import forms
from .models import PatientData

class PatientForm(forms.ModelForm):
    class Meta:
        model = PatientData
        fields = "__all__"  # Include all fields from the PatientData model
