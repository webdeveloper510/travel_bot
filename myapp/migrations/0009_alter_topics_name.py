# Generated by Django 4.2.5 on 2023-10-26 04:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0008_rename_user_id_topics_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='topics',
            name='name',
            field=models.CharField(blank=True, max_length=300, null=True),
        ),
    ]
