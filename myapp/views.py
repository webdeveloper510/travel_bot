
from django.http import HttpResponse
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken
from myapp.renderer import UserRenderer
from myapp.serializers import *
from myapp.models import *
from rest_framework.views import APIView
from distutils import errors
from rest_framework.response import Response
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.permissions import IsAuthenticated,AllowAny
from rest_framework.authentication import TokenAuthentication
from urllib.parse import urljoin
from keras.models import Sequential ,model_from_json
from keras.layers import Embedding , Dense ,GlobalAveragePooling1D
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from django.utils import timezone
from datetime import datetime
import pickle
import requests
import json
import os
import re
import pandas as pd
import numpy as np
import nltk
from polyfuzz import PolyFuzz
from nltk.corpus import stopwords
from TravelBot.settings import DEEP_API_KEY
from autocorrect import Speller
spell=Speller(lang='en')
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
# import nltk
import re
# import string
import pke
from django.shortcuts import get_object_or_404
import sys


# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()


url="http://127.0.0.1:8000/static/media/"
# url="http://16.171.134.22:8000/static/media/"

# Create your views here.

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        # 'refresh': str(refresh),
        'access': str(refresh.access_token),
    }
    
# Api for User Register
class UserRegistrationView(APIView):
    renderer_classes=[UserRenderer]
    def post(self,request,format=None):
        serializer=UserRegistrationSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user=serializer.save()
            return Response({'message':'Registation successful',"status":"status.HTTP_200_OK",'user_id': user.id})
        return Response({errors:serializer.errors},status=status.HTTP_400_BAD_REQUEST)
    
# Api for user login
class UserLoginView(APIView):
    renderer_classes = [UserRenderer]
    def post(self, request, format=None):
        serializer = UserLoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.data.get('email')
        password = serializer.data.get('password')
        user = authenticate(email=email, password=password)
        if user is not None:
            token = get_tokens_for_user(user)
            is_admin = user.is_staff
            return Response({'token':token, "is_admin":is_admin,'msg':'Login Success'}, status=status.HTTP_200_OK)
        else:
            return Response({'errors':{'non_field_errors':['Email or Password is not Valid']}}, status=status.HTTP_404_NOT_FOUND)
  
# Api for user profile
class UserProfileView(APIView):
    renderer_classes = [UserRenderer]
    authentication_classes=[JWTAuthentication]
    def get(self, request, format=None):
        try:
            serializer = UserProfileSerializer(request.user)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Api for user logout
class LogoutUser(APIView):
    renderer_classes = [UserRenderer]
    permission_classes=[IsAuthenticated]
    def post(self, request, format=None):
        return Response({'message':'Logout Successfully','status':'status.HTTP_200_OK'})

# Ai for get user_list
class UserList(APIView):
    permission_classes=[IsAuthenticated]
    authentication_classes=[JWTAuthentication]
    def get(self,request,format=None):
        users = User.objects.filter(is_admin=0).order_by('id').values()
        response=[]
        for user in users:
            created_at=user.get("created_at")
            if created_at:
                user["created_at"]=created_at.strftime('%Y-%m-%d')
            response.append(user)
        return Response({"data":response, "code":200})
        

# Api for upload csv    
class UploadCsv(APIView):
    def post(self, request):
        if 'csv_file' not in request.FILES:
            return Response({"status": status.HTTP_404_NOT_FOUND, "message": "CSV Not Found"})
        
        input_csv = request.FILES['csv_file']
        try:
            data = pd.read_csv(input_csv)
            data.dropna(how="all", inplace=True)
            expected_columns = ["ID","Vendor","NET Cost by Experience", "NET Cost by Hour", "NET Cost Per Person Adult", "NET Cost Per Person Child/Senior", "Is The Guide Included in the cost", "Maximum Pax per cost", "Location", "Description of the Experience", "Time of Visit (in hours)", "Contact First Name", "Contact Last Name", "Contact Number", "Contact Email", "Tag 1", "Tag 2", "Tag 3", "Tag 4", "Tag 5", "Tag 6"]

            # Check if the CSV has the expected columns
            if not all(col in data.columns for col in expected_columns):
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message': 'CSV format is not as expected'})
            
            if not CsvFileData.objects.filter(csvname=input_csv).exists():
                fileupload = CsvFileData.objects.create(csvfile=input_csv, csvname=input_csv.name)
                serializer = CsvFileDataSerializer(fileupload)
                full_url = urljoin(url, str(fileupload.csvfile.name))
                fileupload.csvfile = full_url
                fileupload.save()
                
            for index, row in data.iterrows():
                Vendor = row["Vendor"]
                net_Cost_by_Experience = row["NET Cost by Experience"]
                net_Cost_by_Hour = row["NET Cost by Hour"]
                net_Cost_Per_Person_Adult = row["NET Cost Per Person Adult"]
                net_Cost_Per_Person_Child_Senior = row["NET Cost Per Person Child/Senior"]
                Is_The_Guide_Included_in_the_cost = row["Is The Guide Included in the cost"]
                Maximum_Pax_per_cost = row["Maximum Pax per cost"]
                Location = row["Location"]
                Description_of_the_Experience = row["Description of the Experience"]
                Time_of_Visit_hours = row["Time of Visit (in hours)"]
                Contact_First_Name = row["Contact First Name"]
                Contact_Last_Name = row["Contact Last Name"]
                Contact_Number = row["Contact Number"]
                Contact_Email = row["Contact Email"]
                Tag_1 = row["Tag 1"]
                Tag_2 = row["Tag 2"]
                Tag_3 = row["Tag 3"]
                Tag_4 = row["Tag 4"]
                Tag_5 = row["Tag 5"]
                Tag_6 = row["Tag 6"]

                if not TravelBotData.objects.filter(Vendor = Vendor, net_Cost_by_Experience = net_Cost_by_Experience, net_Cost_by_Hour = net_Cost_by_Hour, net_Cost_Per_Person_Adult = net_Cost_Per_Person_Adult, net_Cost_Per_Person_Child_Senior = net_Cost_Per_Person_Child_Senior, Is_The_Guide_Included_in_the_cost = Is_The_Guide_Included_in_the_cost, Maximum_Pax_per_cost = Maximum_Pax_per_cost, Location = Location, Description_of_the_Experience = Description_of_the_Experience, Time_of_Visit_hours = Time_of_Visit_hours, Contact_First_Name = Contact_First_Name, Contact_Last_Name = Contact_Last_Name, Contact_Number = Contact_Number, Contact_Email = Contact_Email, Tag_1 = Tag_1, Tag_2 = Tag_2, Tag_3 = Tag_3, Tag_4 = Tag_4, Tag_5 = Tag_5, Tag_6 = Tag_6).exists():
                    dataload = TravelBotData.objects.create(Vendor = Vendor, net_Cost_by_Experience = net_Cost_by_Experience, net_Cost_by_Hour = net_Cost_by_Hour, net_Cost_Per_Person_Adult = net_Cost_Per_Person_Adult, net_Cost_Per_Person_Child_Senior = net_Cost_Per_Person_Child_Senior, Is_The_Guide_Included_in_the_cost = Is_The_Guide_Included_in_the_cost, Maximum_Pax_per_cost = Maximum_Pax_per_cost, Location = Location, Description_of_the_Experience = Description_of_the_Experience, Time_of_Visit_hours = Time_of_Visit_hours, Contact_First_Name = Contact_First_Name, Contact_Last_Name = Contact_Last_Name, Contact_Number = Contact_Number, Contact_Email = Contact_Email, Tag_1 = Tag_1, Tag_2 = Tag_2, Tag_3 = Tag_3, Tag_4 = Tag_4, Tag_5 = Tag_5, Tag_6 = Tag_6)
                    dataload.save()
            return Response({'message': "File uploaded and data saved successfully"})
        except Exception as e:
            return Response({'status': status.HTTP_500_INTERNAL_SERVER_ERROR, 'message': str(e)})

    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')     
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS=set(stopwords.words("english"))
        if not text:
            return ""
        text=text.lower()
        text=REPLACE_BY_SPACE_RE.sub(' ',text)
        text=BAD_SYMBOLS_RE.sub(' ',text)
        text=text.replace('x','')
        text=' '.join(word for word in text.split() if word not in STOPWORDS)
        return text
details_dict={
    "hi":"Hello",
    "hey" :"Hi",
    "is anyone there?" :"Hi there ",
    "is anyone there" :"Hi there ",
    "hello" :"Hi",
    "hi there":"hello",
    "how are you" :"I am AI bot. I am always remain well.",
    "bye":"See you later",
    "see you later": "Have a nice day",
    "goodbye":"Bye! Come back again",
    "thanks":"Happy to help!",
    "thank you":"My pleasure",
    "that's helpful":"Any time!",
    "thanks for the help":"You're most welcome!",
    "who are you?":" your bot assistant",
    "could you help me":"Tell me how can assist you",
    "can you help me":"Tell me how can assist you",
    "i need a help":"Tell me your problem to assist you",
    "support me please":"Yes Sure, How can I support you"
}

# Api for predict Answer
class prediction(APIView):
    authentication_classes=[JWTAuthentication]

    def delete(self, request):
        try:
            user = UserProfileSerializer(request.user)
            id = request.data.get('id')
            if not id:
                return Response({'status':status.HTTP_400_BAD_REQUEST, "message":"Please enter id"})
            if not UserActivity.objects.filter(user_id=user.data['id'],id=id).exists():
                return Response({'status':status.HTTP_400_BAD_REQUEST, "message":"Invalid id"})
            chat = UserActivity.objects.filter(user_id=user.data['id'],id=id)
            chat.delete()
            return Response({'status':status.HTTP_200_OK,'message':"Deleted Successfully"})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return Response({'status':  status.HTTP_400_BAD_REQUEST, 'message': str(str(e)+" in line "+str(exc_tb.tb_lineno))})

    def get(self, request):   #to get data from useractivity table with topic id
        try:
            user = UserProfileSerializer(request.user)
            topic_id = request.data.get('topic_id')
            if not topic_id:
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Please enter topic_id'})
            if not UserActivity.objects.filter(user_id=user.data['id'], topic_id_id=topic_id).exists():
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Chat does not exists related to this topic!'})
            data = UserActivity.objects.filter(user_id=user.data['id'], topic_id_id=topic_id).values('user_id','date','questions','answer','topic','topic_id_id')
            for i in data:
                i.update(time=i['date'].time())
                i['date']  = i['date'].date()    
                i['topic_id']  = i['topic_id_id'] 
                del i['topic_id_id']  
            return Response({'status':status.HTTP_200_OK, "message":"Success", 'data': list(data)})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return Response({'status':  status.HTTP_400_BAD_REQUEST, 'message': str(str(e)+" in line "+str(exc_tb.tb_lineno))})

        
    def post(self, request,format=None):
        matched_items = []
        max_non_empty_count = 0
        most_non_empty_list = None
        actual_dict = {}
        AnswerDict = {}
        AnswerList = []
        answer_found = False
        questionInput = request.data.get('query')
        topic_id = request.data.get('topic_id')
        if not topic_id:
            return Response({'status':status.HTTP_404_NOT_FOUND, 'message':"Please enter topic_id"})
        if not Topics.objects.filter(id=topic_id).exists():
            return Response({'status':status.HTTP_404_NOT_FOUND, 'message':"Invalid topic_id"})
        input=spell(questionInput)
        value_found=details_dict.get(input.lower().strip())
        if value_found:
            itenary_answer=value_found
            answer_found = True
        else:
            itenary_answer = None 
            service = TravelBotData.objects.all().order_by('id')
            for i in service:
                newList = [i.Vendor, i.net_Cost_by_Experience, i.net_Cost_by_Hour, i.net_Cost_Per_Person_Adult, i.net_Cost_Per_Person_Child_Senior, i.Is_The_Guide_Included_in_the_cost, i.Maximum_Pax_per_cost, i.Location, i.Description_of_the_Experience, i.Time_of_Visit_hours, i.Contact_First_Name, i.Contact_Last_Name, i.Contact_Number, i.Contact_Email, i.Tag_1, i.Tag_2, i.Tag_3, i.Tag_4, i.Tag_5, i.Tag_6]
                new_genrate_for_test = [element.strip().lower() for element in newList if element != 'nan']
                all_fields = TravelBotData._meta.get_fields()
                field_names = [field.name for field in all_fields]
                del field_names[0]
                count = 0
                for keys in field_names:
                    actual_dict[keys] = newList[count]
                    count += 1  
                resulting_string = '|'.join(new_genrate_for_test)
                pattern = re.compile(rf'{resulting_string}')
                matches = re.findall(pattern, questionInput.lower())
                if matches:
                    if matches not in matched_items:
                        matched_items.append(matches)
                    non_empty_count = sum(1 for item in matches if item.strip())
                    if non_empty_count > max_non_empty_count:
                        max_non_empty_count = non_empty_count
                        most_non_empty_list = matches
                        most_non_empty_list=  [' '.join(word.title() for word in element.split()) for element in most_non_empty_list]
                        for word in most_non_empty_list:
                            if word in newList:
                                AnswerList.append(word)
                                for key , value in actual_dict.items():
                                    for j in AnswerList:
                                        if value == j:
                                            AnswerDict[key] = j
                                    if re.search(key.replace('_', ' ').lower() , questionInput.lower()):
                                        AnswerDict[key] = value
        label = ''
        if AnswerDict:
            r = requests.post("https://api.deepai.org/api/text-generator",{"text":str(AnswerDict)},headers={'api-key':DEEP_API_KEY})
            genratedText = r.json()
            itenary_answer=genratedText['output']
            extractor.load_document(input=itenary_answer, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=10)
            label = keyphrases[0][0]
            assert itenary_answer

            answer_found=True
        if answer_found:
            conversation=UserActivity.objects.create(user_id=request.user.id,questions=questionInput,answer=itenary_answer, topic=label, topic_id_id=topic_id)
            date_time = conversation.date
            datetime_obj = datetime.strptime(str(date_time), "%Y-%m-%d %H:%M:%S.%f%z")  # Use the correct format
            formatted_time = datetime_obj.strftime("%H:%M:%S")
            return Response({"Answer":itenary_answer,"time":formatted_time, "id":conversation.id,'label':conversation.topic},status=status.HTTP_200_OK)
        else:
            return Response({"Answer":"Chatgpt Integeration pending"},status=status.HTTP_204_NO_CONTENT)

class AnswerSuggestion(APIView):
    authentication_classes=[JWTAuthentication]

    def put(self, request, id):
        suggestion=request.data.get('suggestion')
        if UserActivity.objects.get(id=id):
            UserActivity.objects.filter(id=id).update(answer=suggestion)
            return Response({"Result":"Answer's suggestion Updated"},status=status.HTTP_200_OK)
        else:
            return Response({"Result":"Result not found for this question"},status=status.HTTP_204_NO_CONTENT)

class AddQuestionAnswer(APIView):
    authentication_classes=[JWTAuthentication]
    def post(self , request):
        question = request.data.get('question')
        answer = request.data.get('answer')
        extractor.load_document(input=answer, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=10)
        label = keyphrases[0][0]
        createSuggestion = UserActivity.objects.create(user_id = request.user.id ,questions=question, answer=answer, topic=label)
        createSuggestion.save()
        return Response({"data":"Suggestion Added Successfully !"})


# Api for get user history   
class GetUserHistory(APIView):
    authentication_classes=[JWTAuthentication]
    def get(self,request):
        history=UserActivity.objects.filter(user_id=request.user.id).values()
        response=[]
        for user in history:
            date_time=user.get("date")
            if date_time:
                user["time"]=date_time.strftime("%H:%M:%S")
            response.append(user)
        return Response({"data":response},status=status.HTTP_200_OK)
    
# Api for get CSV files
class GetAlluploadedcsv(APIView):
    authentication_classes=[JWTAuthentication]
    def get(self,request):
        history=CsvFileData.objects.all().values().order_by("id")
        return Response({"data":history,"code":200})

# Api for user delete
class DeleteUser(APIView):
    def delete(self, request, id):
        try:
            user = User.objects.get(id=id)
            user.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            return Response({'status':  status.HTTP_204_NO_CONTENT, 'message': str(e)})
        
# APi For Profile Update
class ProfileUpdate(APIView):
    authentication_classes=[JWTAuthentication]
    def put(self,request,id):
        try:
            user_update = User.objects.get(id = id)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
        serializer=UpdateProfileSerializer(user_update , data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,status=status.HTTP_200_OK)
        return Response(serializer.errors,status=status.HTTP_400_BAD_REQUEST)

        
class ActiveInactive(APIView):

    def post(self, request, id):
        user_active = request.data.get("is_active")
        user = User.objects.filter(id=id).update(is_active=user_active)
        user = User.objects.get(id=id)  # Fetch the user object again after updating
        if user.is_active:
            return Response({"message": "User is Active"}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "User is Inactive"}, status=status.HTTP_200_OK)
        
        
class UserChangePasswordView(APIView):
  renderer_classes = [UserRenderer]
  authentication_classes=[JWTAuthentication]
  def post(self, request, format=None):
    serializer = UserChangePasswordSerializer(data=request.data, context={'user':request.user})
    serializer.is_valid(raise_exception=True)
    return Response({'msg':'Password Changed Successfully'}, status=status.HTTP_200_OK)

    
class CsvDeleteView(APIView):
    authentication_classes=[JWTAuthentication]
    def delete(self, request, id):
        try:
            csvfile = CsvFileData.objects.get(id=id)
            csvfile.delete()
            return Response({"message":"File Delete Successfully"},status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            return Response({'status':  status.HTTP_204_NO_CONTENT, 'message': str(e)})

class TopicsView(APIView):
    authentication_classes=[JWTAuthentication]

    def post(self, request):
        try:
            user = UserProfileSerializer(request.user)
            name = request.data.get('name')
            if not name:
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Please enter name'})
            data = Topics.objects.create(user_id=user.data['id'], name=name)
            return Response({'status':status.HTTP_204_NO_CONTENT, "message":"Success", 'data': {'id':data.id, 'name':data.name, 'user_id':data.user_id}})
        except Exception as e:
            return Response({'status':  status.HTTP_204_NO_CONTENT, 'message': str(e)})

    def get(self, request):   #to get topics list of user
        try:
            user = UserProfileSerializer(request.user)
            data = Topics.objects.filter(user_id=user.data['id']).values('id','user_id','name')
            return Response({'status':status.HTTP_204_NO_CONTENT, "message":"Success", 'data': list(data)})
        except Exception as e:
            return Response({'status':  status.HTTP_204_NO_CONTENT, 'message': str(e)})
        
    def put(self, request):   
        try:
            topic_id = request.data.get('topic_id')
            name = request.data.get('name')
            if not topic_id:
                return Response({'status':status.HTTP_400_BAD_REQUEST, "message":"Please enter topic_id"})
            if not Topics.objects.filter(id=topic_id).exists():
                return Response({'status':status.HTTP_400_BAD_REQUEST, "message":"Invalid topic_id"})
            user = UserProfileSerializer(request.user)
            Topics.objects.filter(id=topic_id,user_id=user.data['id']).update(name=name)
            data = get_object_or_404(Topics, id=topic_id)
            return Response({'status':status.HTTP_204_NO_CONTENT, "message":"Success", 'data': {'id':data.id, 'user_id':data.user_id, 'name':data.name}})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return Response({'status':  status.HTTP_400_BAD_REQUEST, 'message': str(str(e)+" in line "+str(exc_tb.tb_lineno))})

