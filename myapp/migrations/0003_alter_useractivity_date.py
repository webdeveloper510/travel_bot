# Generated by Django 4.2.5 on 2023-10-18 07:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_useractivity'),
    ]

    operations = [
        migrations.AlterField(
            model_name='useractivity',
            name='date',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]