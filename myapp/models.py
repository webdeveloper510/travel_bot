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

'''
Vendor
NET Cost by Experience
NET Cost by Hour
NET Cost Per Person Adult
NET Cost Per Person Child/Senior
Is The Guide Included in the cost
Maximum Pax per cost
Location
Description of the Experience
Time of Visit (in hours)
Contact First Name
Contact Last Name
Contact Number
Contact Email
Tag 1
Tag 2
Tag 3
Tag 4
Tag 5
Tag 6

'''

class TravelBotData(models.Model):
    # topic_name=models.CharField(max_length=200,null=True,blank=True)
    # question=models.TextField(max_length=1000)
    # answer=models.TextField(max_length=1000)
    
    Vendor = models.CharField(max_length=255, null=True , blank=True)
    net_Cost_by_Experience = models.CharField(max_length=255, null=True , blank=True)
    net_Cost_by_Hour = models.CharField(max_length=255, null=True , blank=True)
    net_Cost_Per_Person_Adult = models.CharField(max_length=255, null=True , blank=True)
    net_Cost_Per_Person_Child_Senior = models.CharField(max_length=255, null=True , blank=True)
    Is_The_Guide_Included_in_the_cost = models.CharField(max_length=255, null=True , blank=True)
    Maximum_Pax_per_cost = models.CharField(max_length=255, null=True , blank=True)
    Location = models.CharField(max_length=255, null=True , blank=True)
    Description_of_the_Experience = models.CharField(max_length=255, null=True , blank=True)
    Time_of_Visit_hours = models.CharField(max_length=255, null=True , blank=True)
    Contact_First_Name = models.CharField(max_length=255, null=True , blank=True)
    Contact_Last_Name = models.CharField(max_length=255, null=True , blank=True)
    Contact_Number = models.CharField(max_length=255, null=True , blank=True)
    Contact_Email = models.CharField(max_length=255, null=True , blank=True)
    Tag_1 = models.CharField(max_length=255, null=True , blank=True)
    Tag_2 = models.CharField(max_length=255, null=True , blank=True)
    Tag_3 = models.CharField(max_length=255, null=True , blank=True)
    Tag_4 = models.CharField(max_length=255, null=True , blank=True)
    Tag_5 = models.CharField(max_length=255, null=True , blank=True)
    Tag_6 = models.CharField(max_length=255, null=True , blank=True)

class CsvFileData(models.Model):
    csvfile=models.FileField(upload_to="user_csv/",blank=True,null=True)
    csvname=models.CharField(max_length=200,blank=True,null=True)
    
class Topics(models.Model):
    user=models.ForeignKey(User , on_delete=models.CASCADE)
    name=models.CharField(max_length=300,blank=True,null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    vendor_name = models.TextField(blank=True, null=True)
    
class UserActivity(models.Model):
    user=models.ForeignKey(User , on_delete=models.CASCADE)
    date=models.DateTimeField(auto_now_add=True)
    questions=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)
    topic=models.TextField(max_length=1000)
    topic_id=models.ForeignKey(Topics , on_delete=models.CASCADE)


class UserDetailGethringForm(models.Model):
    SelectBudget = [
        ('Up to €5000','Up to €5000'),
        ('Between €5000 - €10,000','Between €5000 - €10,000'),
        ('Between €10,000 - €15,000','Between €10,000 - €15,000'),
        ('Between €15,000 - €20,000','Between €15,000 - €20,000'),
        ('€20,000 and over','€20,000 and over'),
        ('Other','Other'),
    ]

    Malta_Experience = [
        ('Not more than 4 Hours','Not more than 4 Hours'),
        ('Not more than 6 Hours','Not more than 6 Hours'),
        ('Not more than 8 Hours','Not more than 8 Hours'),
    ]
    Start_Time = [
        ('08:30','08:30'),
        ('09:00','09:00'),
        ('09:30','09:30'),
        ('10:00','10:00'),
        ('Other','Other'),
    ]
    Lunch_Time = [
        ('12:00','12:00'),
        ('12:30','12:30'),
        ('13:00','13:00'),
        ('13:30','13:30'),
        ('Other','Other'),

    ]
    Dinner_Time = [
        ('19:00','19:00'),
        ('19:30','19:30'),
        ('20:00','20:00'),
        ('20:30','20:30'),
        ('Other','Other'),
    ]

    user=models.ForeignKey(User , on_delete=models.CASCADE)
    employee_name=models.CharField(max_length=255)
    numberOfTour=models.IntegerField(unique=True)
    client_firstName=models.CharField(max_length=255)
    client_lastName=models.CharField(max_length=255)
    nationality=models.CharField(max_length=255)
    datesOfTravel=models.CharField(max_length=255)
    numberOfTravellers=models.IntegerField()
    agesOfTravellers=models.CharField(max_length=255)
    select_budget=models.TextField(choices=SelectBudget)
    flightArrivalTime=models.TextField(blank=True ,null=True)
    flightArrivalNumber=models.CharField(max_length=255,blank=True,null=True)
    flightDepartureTime=models.TextField(blank=True , null=True)
    flightDepartureNumber=models.CharField(max_length=255,blank=True,null=True)
    accommodation_specific=models.TextField()
    malta_experience=models.TextField(choices=Malta_Experience)
    start_time=models.TextField(choices=Start_Time)
    lunch_time=models.TextField(choices=Lunch_Time)
    dinner_time=models.TextField(choices=Dinner_Time)
    issues_n_phobias=models.TextField()
    other_details=models.TextField(blank=True, null=True)

