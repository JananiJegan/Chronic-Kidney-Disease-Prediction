# Generated by Django 4.2.6 on 2024-09-25 17:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PatientData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.IntegerField()),
                ('gender', models.IntegerField()),
                ('bmi', models.FloatField()),
                ('systolic_bp', models.IntegerField()),
                ('diastolic_bp', models.IntegerField()),
                ('fasting_blood_sugar', models.IntegerField()),
                ('hba1c', models.FloatField()),
                ('serum_creatinine', models.FloatField()),
                ('gfr', models.IntegerField()),
                ('smoking', models.BooleanField()),
            ],
        ),
    ]
