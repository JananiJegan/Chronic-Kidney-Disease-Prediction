# Generated by Django 4.1 on 2024-09-28 14:20

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("Users", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="patientdata",
            name="prediction",
            field=models.CharField(default="default_value", max_length=255),
        ),
        migrations.AlterField(
            model_name="patientdata",
            name="gender",
            field=models.CharField(max_length=1),
        ),
        migrations.AlterField(
            model_name="patientdata",
            name="gfr",
            field=models.FloatField(),
        ),
    ]
