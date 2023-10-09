from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from .models import *

class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model=User
        fields=['id','email','password','firstname','lastname']

        extra_kwargs={
            'email': {'error_messages': {'required': "email is required",'blank':'please provide a email'}},
            'password': {'error_messages': {'required': "password is required",'blank':'please Enter a password'}},
            'firstname': {'error_messages': {'required': "firstname is required",'blank':'firstname could not blank'}},
            'lastname': {'error_messages': {'required': "lastname is required",'blank':'lastname could not blank'}},
          }
        
    def create(self, validated_data,):
       user=User.objects.create(
       email=validated_data['email'],
       firstname=validated_data['firstname'],
       lastname=validated_data['lastname'],)
       user.set_password(validated_data['password']) 
       user.save()
       return user
   
class UserLoginSerializer(serializers.ModelSerializer):
  email = serializers.EmailField(max_length=255)
  class Meta:
    model = User
    fields = ['email', 'password']
    
class UserProfileSerializer(serializers.ModelSerializer):
  class Meta:
    model = User
    fields = ['id', 'email', 'firstname','lastname']
    


class TravelBotDataSerializer(serializers.ModelSerializer):
    class Meta:
        model= TravelBotData
        fields = '__all__'
           
    def create(self, validate_data):
        return TravelBotData.objects.create(**validate_data)
    
class CsvFileDataSerializer(serializers.ModelSerializer):
    class Meta:
        model=CsvFileData
        fields="__all__"
        
    def create (self,validate_data):
        return CsvFileData.objects.create(**validate_data)
    
class UpdateProfileSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True,required=False)
    
    # for password encrypting format 
    def update(self, instance, validated_data):
        password = validated_data.get('password',None)
        if password:
            instance.password = validated_data.get('password', instance.password)
            validated_data['password'] = make_password(instance.password) 
        else:
            validated_data.pop('password', None)
        
        return super(UpdateProfileSerializer , self).update(instance,validated_data)

    class Meta:
        model=User
        fields=["id","firstname","lastname","email","password"]
        
        extra_kwargs = {
        'firstname': {'required': False},
        'lastname':{'required': False},
        'email': {'required': False},
        'password': {'required': False},
        
        }
        def validate_password(self,password):
        
            if len(password)< 8:
                raise serializers.ValidationError("Password must be more than 8 character.")
            if not any(char.isdigit() for char in password):
                raise serializers.ValidationError('Password must contain at least one digit.')
            return password
            
    
class UserChangePasswordSerializer(serializers.Serializer):
  password = serializers.CharField(max_length=255, style={'input_type':'password'}, write_only=True)
  password2 = serializers.CharField(max_length=255, style={'input_type':'password'}, write_only=True)
  class Meta:
    fields = ['password', 'password2']

  def validate(self, attrs):
    password = attrs.get('password')
    password2 = attrs.get('password2')
    user = self.context.get('user')
    if password != password2:
      raise serializers.ValidationError("Password and Confirm Password doesn't match")
    user.set_password(password)
    user.save()
    return attrs
            
