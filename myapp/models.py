from django.db import models
from django.contrib.auth.models import *

# Create your models here.

class UserManager(BaseUserManager):
    def create_user(self, email, firstname, lastname, password=None):
        
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            firstname=firstname,
            lastname=lastname,
                         )

        user.set_password(password)
        user.save(using=self._db)
        return user
    def create_superuser(self, email, password=None):
       
        user = self.create_user(
            email,
            firstname= "None",  
            lastname= "None",
            password=password,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user

#  Custom User Model
class User(AbstractBaseUser,PermissionsMixin):
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    firstname = models.CharField(max_length=80)
    lastname = models.CharField(max_length=80)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    objects = UserManager()

    USERNAME_FIELD = 'email'

    REQUIRED_FIELDS = []


    def ___str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        return self.is_admin


    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        return self.is_admin
    

# class Topic(models.Model):
#     topic_name=models.CharField(max_length=200,null=True,blank=True)

class TravelBotData(models.Model):
    topic_name=models.CharField(max_length=200,null=True,blank=True)
    question=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)
    
    
class CsvFileData(models.Model):
    csvfile=models.FileField(upload_to="user_csv/",blank=True,null=True)
    csvname=models.CharField(max_length=200,blank=True,null=True)
    
    
class UserActivity(models.Model):
    user=models.ForeignKey(User , on_delete=models.CASCADE)
    date=models.DateTimeField(auto_now_add=True)
    questions=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)
    
