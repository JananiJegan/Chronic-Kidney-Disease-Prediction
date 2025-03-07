from django.db import models

# Create your models here.
class PatientData(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=1)
    bmi = models.FloatField()
    systolic_bp = models.IntegerField()
    diastolic_bp = models.IntegerField()
    fasting_blood_sugar = models.IntegerField()
    hba1c = models.FloatField()
    serum_creatinine = models.FloatField()
    gfr = models.FloatField()
    smoking = models.BooleanField()
    prediction = models.CharField(max_length=255, default='default_value')  # Change to your desired default
    
    def __str__(self):
        return f"Patient {self.id} - Age: {self.age}, Gender: {self.gender}"

