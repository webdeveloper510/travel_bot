# Generated by Django 4.2.6 on 2023-11-28 10:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0010_useractivity_topic_id_alter_useractivity_topic'),
    ]

    operations = [
        migrations.AddField(
            model_name='topics',
            name='vendor_name',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
