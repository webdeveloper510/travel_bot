
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
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()
import random
num_random_rows = 1
from keybert import KeyBERT
import sys
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, CSVLoader, DataFrameLoader 
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma 
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import ast
load_dotenv()


# url="http://127.0.0.1:8000/static/media/"
url="http://16.170.254.147:8000/static/media/"

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
    "who are you":"I am AI bot.",   
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
    
    
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')     
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS=set(stopwords.words("english"))
        if not text:
            return ""
        text=text.lower()
        text=REPLACE_BY_SPACE_RE.sub(' ',text)
        text=BAD_SYMBOLS_RE.sub(' ',text)
        # text=text.replace('x','')
        text=' '.join(word for word in text.split() if word not in STOPWORDS)
        return text
    
    def automaticgetlabel(self, text):
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text)
        keyword= kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english',top_n=1)
        return keyword

    def find_best_header_match(self,unique_results_list, inputlist):
        best_match = []
        if unique_results_list:
            for each_rec in unique_results_list:
                for split_item in inputlist:
                    similarity = fuzz.ratio(each_rec.lower(), split_item.lower())
                    if similarity >= 90:    
                        best_match.append(each_rec)

        if unique_results_list:
            max_common_items = 0  # Initialize maximum common items
            header_with_max_common = None  # Initialize the header with the most common items

            for header in unique_results_list:
                header_words = header.lower().split()
                common_words = set(header_words).intersection(inputlist)
                common_items_count = len(common_words)
                if len(inputlist) >= 2:  # Check the length of inputlist
                    if common_items_count >= 2:
                        if common_items_count > max_common_items:
                            max_common_items = common_items_count
                            header_with_max_common = header
                            best_match.append(header_with_max_common)

                elif len(inputlist) == 1:  # Run the elif condition when inputlist length is 1
                    if common_items_count == 1:
                        best_match.append(header)

            if header_with_max_common is not None and len(inputlist) >= 2:
                best_match.append(header_with_max_common)

        return list(set(best_match)) # Return the best_match list
    
    def find_best_label_matches(self, dictionary_list, input_list):
            matches = []
            numberToCheck = 0 
            indexToCheck = None
            indexesList = []
            SelectedGetQueryDicts = []
            for i, data in enumerate(dictionary_list):
                for key, value in data.items():
                    valueList = value.lower().split(" ")
                    newInput =  [l for l in input_list if l.lower()!="visit" and l.lower()!="experience"  and l.lower()!="guide"]
                    current_matches = [j for j in newInput if j.lower() in valueList]
                    if len(current_matches) > numberToCheck:
                        numberToCheck = len(current_matches)
                        indexToCheck = i
                        matches = current_matches
                    if len(current_matches) > 2:
                        numberToCheck = len(current_matches)
                        indexToCheck = i
                        matches = current_matches
                    if indexToCheck not in indexesList and indexToCheck!=None:
                        indexesList.append(indexToCheck)
 
            if indexesList is not None:    
                for ij in indexesList:
                    SelectedGetQueryDicts.append(dictionary_list[ij])
                return SelectedGetQueryDicts
            else:
                return 'None'

    def post(self, request,format=None):
        answer_found = False
        itenary_answer = None
        label = ''            
        unique_results = set()
        actual_list = []
        dictionary_list = []
        queryValue_dict = None
        VendorNameDict = {}
        
        # =================================================================

        questionInput = request.data.get('query')
        topic_id = request.data.get('topic_id')
        vendor_get=request.data.get('vendor_name')
        correct_input = self.clean_text(questionInput)
        inputlist = correct_input.split(" ")
        print(vendor_get , "****************************************************************")
        # =================================================================

        if not Topics.objects.filter(user_id=request.user.id).exists():
            data = Topics.objects.create(user_id=request.user.id, name=questionInput)
            topic_id = data.id
        else:
            if not topic_id:
                return Response({'status':status.HTTP_404_NOT_FOUND, 'message':"Please enter topic_id"})
            if not Topics.objects.filter(id=topic_id).exists():
                return Response({'status':status.HTTP_404_NOT_FOUND, 'message':"Invalid topic_id"})
            
        # =================================================================
 
        input=spell(questionInput)
        value_found=details_dict.get(input.lower().strip())
        if value_found:
            itenary_answer=value_found
            answer_found = True 
        else:
            answer_found = False 
            SelectedVendorData = None
            service = TravelBotData.objects.all().order_by('id')
            for i in service:
                actual_dict = {}
                newList = [i.Vendor, i.net_Cost_by_Experience, i.net_Cost_by_Hour, i.net_Cost_Per_Person_Adult, i.net_Cost_Per_Person_Child_Senior, i.Is_The_Guide_Included_in_the_cost, i.Maximum_Pax_per_cost, i.Location, i.Description_of_the_Experience, i.Time_of_Visit_hours, i.Contact_First_Name, i.Contact_Last_Name, i.Contact_Number, i.Contact_Email, i.Tag_1, i.Tag_2, i.Tag_3, i.Tag_4, i.Tag_5, i.Tag_6]
                # ALL columns name get from database
                all_fields = TravelBotData._meta.get_fields()
                field_names = [field.name for field in all_fields]
                del field_names[0]

                # Create a dictionary with column name and its values without none value.

                count = 0
                for keys in field_names:
                    keys_replace = keys.replace("_", " ")
                    if newList[count] != "nan" and newList[count] != " ":
                        actual_dict[keys_replace] = newList[count]
                        actual_list.append(keys_replace)
                    count += 1
                dictionary_list.append(actual_dict)
            actual_keys = list(set(actual_list))

            for items in inputlist:
                for key in actual_keys:
                    assert items
                    if items in key.lower():
                        unique_results.add(key)
            
            queryValue_dict=self.find_best_label_matches(dictionary_list,inputlist)
            print(queryValue_dict)
        # =================================================================

            if queryValue_dict == []:
                VendorSelectList = []
                print("here , =================================================================")
                if vendor_get:
                    vendor_select = json.loads(vendor_get)
                  
                    for uniqueVendor in vendor_select['vendor_name']:
                        SelectedVendorData = TravelBotData.objects.filter(Vendor=uniqueVendor).values()[0]
                        modified_keys_data = {key.replace("_", " "): value for key, value in SelectedVendorData.items()}
                        if modified_keys_data not in VendorSelectList:
                            VendorSelectList.append(modified_keys_data)
                queryValue_dict = VendorSelectList       
                VendorSelectList = []
        # =================================================================

        if queryValue_dict:
            print(queryValue_dict)
            newList = []
            for everyDict in queryValue_dict:
                newList.append(everyDict['Vendor'])
            VendorNameDict["vendor_name"] = newList
            df = pd.DataFrame(queryValue_dict)
            df.to_csv(f'output{request.user.id}.csv', index=False)
            
            os.environ["OPENAI_API_KEY"] = os.getenv("chat_key")
            pathvar = f'output{request.user.id}.csv'
            print(pathvar)
            postPrompt = "Do not give me any information that is not mentioned in the PROVIDED CONTEXT."
            loader = CSVLoader(pathvar)
            index = VectorstoreIndexCreator().from_loaders([loader])
            itenary_answer = index.query(questionInput + " " + postPrompt, llm=ChatOpenAI())
            extractor.load_document(input=itenary_answer, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=10)
            if keyphrases:
                label = keyphrases[0][0]
            answer_found = True

        # =================================================================

        if answer_found:         
            conversation=UserActivity.objects.create(user_id=request.user.id,questions=questionInput,answer=itenary_answer, topic=label, topic_id_id=topic_id)
            date_time = conversation.date
            datetime_obj = datetime.strptime(str(date_time), "%Y-%m-%d %H:%M:%S.%f%z")  # Use the correct format
            formatted_time = datetime_obj.strftime("%H:%M:%S")
            if Topics.objects.filter(id=topic_id):
                print(Topics.objects.filter(id=topic_id).values(), "================================")
            return Response({"Answer":itenary_answer,"time":formatted_time, "id":conversation.id,'label':conversation.topic,"vendor_name":VendorNameDict},status=status.HTTP_200_OK)
        else:
            conversation=UserActivity.objects.create(user_id=request.user.id,questions=questionInput,answer="Data not found !! I am in learning Stage.", topic=label, topic_id_id=topic_id)
            return Response({"Answer":"Data not found !! I am in learning Stage."},status=status.HTTP_400_BAD_REQUEST)



class ChatDetailsByID(APIView):
    authentication_classes=[JWTAuthentication]

    def get(self, request , topic_id):   #to get data from useractivity table with topic id
        try:
            user = UserProfileSerializer(request.user)
            if not topic_id:
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Please enter topic_id'})
            if not UserActivity.objects.filter(user_id=user.data['id'], topic_id_id=topic_id).exists():
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Chat does not exists related to this topic!'})
            data = UserActivity.objects.filter(user_id=user.data['id'], topic_id_id=topic_id).values('id','user_id','date','questions','answer','topic','topic_id_id')
            for i in data:
                i.update(time=i['date'].time())
                i['date']  = i['date'].date()    
                i['topic_id']  = i['topic_id_id'] 
                del i['topic_id_id']  
            return Response({'status':status.HTTP_200_OK, "message":"Success", 'data': list(data)})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return Response({'status':  status.HTTP_400_BAD_REQUEST, 'message': str(str(e)+" in line "+str(exc_tb.tb_lineno))})


class AnswerSuggestion(APIView):
    authentication_classes=[JWTAuthentication]

    def put(self, request, id):
        suggestion=request.data.get('suggestion')
        if UserActivity.objects.get(id=id):
            UserActivity.objects.filter(id=id).update(answer=suggestion)
            return Response({"Result":"Answer's suggestion Updated"},status=status.HTTP_200_OK)
        else:
            return Response({"Result":"Result not found for this question"},status=status.HTTP_204_NO_CONTENT)




class ChatDetailsByID(APIView):
    authentication_classes=[JWTAuthentication]

    def get(self, request , topic_id):   #to get data from useractivity table with topic id
        try:
            user = UserProfileSerializer(request.user)
            # topic_id = request.data.get('topic_id')
            # print(topic_id)
            if not topic_id:
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Please enter topic_id'})
            if not UserActivity.objects.filter(user_id=user.data['id'], topic_id_id=topic_id).exists():
                return Response({'status': status.HTTP_400_BAD_REQUEST, 'message':'Chat does not exists related to this topic!'})
            data = UserActivity.objects.filter(user_id=user.data['id'], topic_id_id=topic_id).values('id','user_id','date','questions','answer','topic','topic_id_id',)
            for i in data:
                i.update(time=i['date'].time())
                i['date']  = i['date'].date()    
                i['topic_id']  = i['topic_id_id'] 
                del i['topic_id_id']  
            return Response({'status':status.HTTP_200_OK, "message":"Success", 'data': list(data)})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return Response({'status':  status.HTTP_400_BAD_REQUEST, 'message': str(str(e)+" in line "+str(exc_tb.tb_lineno))})


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
        topic_id = request.data.get('topic_id')
        if not Topics.objects.filter(user_id=request.user.id).exists():
            data = Topics.objects.create(user_id=request.user.id, name='admin suggestion')
            topic_id = data.id
        else:
            if not topic_id:
                return Response({"status":status.HTTP_400_BAD_REQUEST, "message":"Please enter topic_id"})
            if not Topics.objects.filter(id=topic_id).exists():
                return Response({"status":status.HTTP_400_BAD_REQUEST, "message":"topic_id does not exists!"})
        extractor.load_document(input=answer, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=10)
        label = keyphrases[0][0]
        createSuggestion = UserActivity.objects.create(user_id = request.user.id ,questions=question, answer=answer, topic=label , topic_id_id = topic_id)
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
            return Response({'status':status.HTTP_204_NO_CONTENT, "message":"Success", 'data': {'id':data.id, 'name':data.name, 'user_id':data.user_id, "created_at":data.created_at.date()}})
        except Exception as e:
            return Response({'status':  status.HTTP_204_NO_CONTENT, 'message': str(e)})

    def get(self, request):   #to get topics list of user
        try:
            user = UserProfileSerializer(request.user)
            data = Topics.objects.filter(user_id=user.data['id']).values('id','user_id','name' , 'created_at__date', "vendor_name")
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

class UpdateTopicsView(APIView):
    authentication_classes=[JWTAuthentication]
    def put(self, request, topic_id):   
        try:
            # topic_id = request.data.get('topic_id')
            name = request.data.get('name')
            if not topic_id:
                return Response({'status':status.HTTP_400_BAD_REQUEST, "message":"Please enter topic_id"})
            if not Topics.objects.filter(id=topic_id).exists():
                return Response({'status':status.HTTP_400_BAD_REQUEST, "message":"Invalid topic_id"})
            # user = UserProfileSerializer(request.user)
            Topics.objects.filter(id=topic_id,user_id=request.user.id).update(name=name)
            data = get_object_or_404(Topics, id=topic_id)
            return Response({'status':status.HTTP_204_NO_CONTENT, "message":"Success", 'data': {'id':data.id, 'user_id':request.user.id, 'name':data.name}})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return Response({'status':  status.HTTP_400_BAD_REQUEST, 'message': str(str(e)+" in line "+str(exc_tb.tb_lineno))})