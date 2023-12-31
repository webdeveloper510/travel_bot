# Generated by Django 4.2.5 on 2023-11-22 05:18

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('email', models.EmailField(max_length=255, unique=True, verbose_name='email address')),
                ('firstname', models.CharField(max_length=80)),
                ('lastname', models.CharField(max_length=80)),
                ('is_active', models.BooleanField(default=True)),
                ('is_admin', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.group', verbose_name='groups')),
                ('user_permissions', models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.permission', verbose_name='user permissions')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='CsvFileData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('csvfile', models.FileField(blank=True, null=True, upload_to='user_csv/')),
                ('csvname', models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Topics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=300, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('vendor_name', models.CharField(blank=True, max_length=255, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='TravelBotData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Vendor', models.CharField(blank=True, max_length=255, null=True)),
                ('net_Cost_by_Experience', models.CharField(blank=True, max_length=255, null=True)),
                ('net_Cost_by_Hour', models.CharField(blank=True, max_length=255, null=True)),
                ('net_Cost_Per_Person_Adult', models.CharField(blank=True, max_length=255, null=True)),
                ('net_Cost_Per_Person_Child_Senior', models.CharField(blank=True, max_length=255, null=True)),
                ('Is_The_Guide_Included_in_the_cost', models.CharField(blank=True, max_length=255, null=True)),
                ('Maximum_Pax_per_cost', models.CharField(blank=True, max_length=255, null=True)),
                ('Location', models.CharField(blank=True, max_length=255, null=True)),
                ('Description_of_the_Experience', models.CharField(blank=True, max_length=255, null=True)),
                ('Time_of_Visit_hours', models.CharField(blank=True, max_length=255, null=True)),
                ('Contact_First_Name', models.CharField(blank=True, max_length=255, null=True)),
                ('Contact_Last_Name', models.CharField(blank=True, max_length=255, null=True)),
                ('Contact_Number', models.CharField(blank=True, max_length=255, null=True)),
                ('Contact_Email', models.CharField(blank=True, max_length=255, null=True)),
                ('Tag_1', models.CharField(blank=True, max_length=255, null=True)),
                ('Tag_2', models.CharField(blank=True, max_length=255, null=True)),
                ('Tag_3', models.CharField(blank=True, max_length=255, null=True)),
                ('Tag_4', models.CharField(blank=True, max_length=255, null=True)),
                ('Tag_5', models.CharField(blank=True, max_length=255, null=True)),
                ('Tag_6', models.CharField(blank=True, max_length=255, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserActivity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('questions', models.TextField(max_length=1000)),
                ('answer', models.TextField(max_length=1000)),
                ('topic', models.TextField(max_length=1000)),
                ('topic_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='myapp.topics')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
