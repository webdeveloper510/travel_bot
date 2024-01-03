
from django.http import HttpResponse
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken
from myapp.renderer import UserRenderer
from myapp.serializers import *
from myapp.tests import *
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
from datetime import datetime,timedelta
import re
import pandas as pd
import numpy as np
import openai
from polyfuzz import PolyFuzz
from nltk.corpus import stopwords
from TravelBot.settings import GOOGLE_MAPS_KEY
from autocorrect import Speller
spell=Speller(lang='en')
from fuzzywuzzy import fuzz
from langchain.chat_models import ChatOpenAI
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
import googlemaps
from dotenv import load_dotenv
load_dotenv()
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import CSVLoader
import ast
import requests
from TravelBot.settings import DEEP_API_KEY
from jinja2 import Template
import spacy
from django.db.models import Q
nlp=spacy.load("en_core_web_sm")
from bs4 import BeautifulSoup
# url="http://127.0.0.1:8000/static/media/"
url="http://16.170.254.147:8000/static/media/"
# Create your views\ here.

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


class Prediction(APIView):
    authentication_classes=[JWTAuthentication]
    
    def contains_word(self, s, w):
     return (' ' + w + ' ') in (' ' + s + ' ')
    
    # Function for user query cleaning and preprocessing 
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')     
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS=set(stopwords.words("english"))
        if not text:
            return ""
        text=text.lower()
        text=REPLACE_BY_SPACE_RE.sub(' ',text)
        text=BAD_SYMBOLS_RE.sub(' ',text)
        text=' '.join(word for word in text.split() if word not in STOPWORDS)
        return text
   
    
    # function for getting the synonym from user query
    def find_synonyms(self,text):
        from nltk.corpus import wordnet
        synonyms_list=list(set([l.name() for syn in wordnet.synsets(text) for l in syn.lemmas()]))
        return synonyms_list
    
    # Function for match csv values (SQL DATABASE)
    def find_vendor_values(self, dictionary_list, input_list):
        inputlist = list(set(input_list) - {"visit","experience","palace"})
        header_vendor_text = "Vendor"
        result_list = {}
        index = 0
        countWords = 0 
        for d in dictionary_list:
            for word in inputlist:
                if header_vendor_text in d and self.contains_word(d[header_vendor_text].lower().replace('hour', '').strip(),word.lower()):
                    countWords = countWords + 1
            result_list[str(index)] = countWords
            index = index + 1
            countWords = 0 
        
        # filter dict based on true value
        filtered_dict = {key: value for key, value in result_list.items() if value != 0}
        # Sort dictionary based on the 
        sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_indices = list(sorted_dict.keys())
        filtered_dictionary_list = [dictionary_list[int(index)] for index in sorted_indices]
        return filtered_dictionary_list
   
    # Function for getting value from tag and location column
    def find_tags_values(self, dictionary_list, input_list):
        print("input_list=======================>>>",input_list)
        matches_row_first_condition = []  # Initialize an empty list to store matches
        matches_row_second_condition=[]
        for  data in dictionary_list:
            new_dict = {
                'Location': data.get('Location', ''),
                'Tag 1': data.get('Tag 1', ''),
                'Tag 2': data.get('Tag 2', ''),
                'Tag 3': data.get('Tag 3', ''),
                'Tag 4': data.get('Tag 4', ''),
                'Tag 5': data.get('Tag 5', ''),
                'Tag 6': data.get('Tag 6', '')
            }
            for key , value in new_dict.items():
                values=value.lower().split(" ")
                current_matches = [word for word in input_list if word in values]
                if current_matches:
                    if len(set(map(len,current_matches)))==1:
                        matches_row_first_condition.append(data)
                        break
                    elif any(len(matches) >=2 for matches in current_matches):
                        matches_row_second_condition.append(data)
                        break
    
        if matches_row_first_condition:
            return matches_row_first_condition
        elif matches_row_second_condition:
            return matches_row_second_condition
        else:
            return "None"
    
    # function to get intent from user query to check that user ask about whch column data
    def get_header_name(self, unique_results, inputlist):
        inputlist = list(set(inputlist) - {"visit"})
        result_list = {}
        for index, tag in enumerate(unique_results):
            count_words = 0
            for word in inputlist:
                matched_header = self.contains_word(tag.lower(), word)
                count_words += 1 if matched_header else 0  # Increment count if the word is present
            
            result_list[str(index)] = count_words
        filtered_dict = {key: value for key, value in result_list.items() if value != 0}
        # Sort dictionary based on the 
        sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_indices = list(sorted_dict.keys())
        filtered_header= [unique_results[int(index)] for index in sorted_indices]
        return filtered_header
    
    # function for check google ask by google maps
    def findMapIntent(self,inputlist):
        PropertiesMatchMaps = []
        GMapsProperties = ["Direction","Navigate","Route","Driving directions","Walking directions","Turn navigation","Waypoints",
                            "Public transportation","Points of interest","What's near","Traffic","GPS (Global Positioning System)","GPS","Landmarks",
                            "Estimated time","Traffic jam","Distance","far","Duration","Time to","Real-time traffic","Specific","Maps"
                            "Map of","Geocoding:","Latitude","longitude","coordinates","Street","Calculate","Travel",
                            "Alternative","Other ways to","Different route"]
        for each_rec in GMapsProperties:
            for split_item in inputlist:
                similarity = fuzz.ratio(each_rec.lower(), split_item.lower())
                if similarity >= 65:    
                    PropertiesMatchMaps.append(each_rec)
        return PropertiesMatchMaps
    
    # function for get the distance between two location with three mode
    def Calcuate_distance_time(self,origin_direction ,destination,mode):
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)
        directions_result = gmaps.directions(origin_direction, destination, mode)
        if directions_result:
            distance=directions_result[0]['legs'][0]['distance']['text']
            duration=directions_result[0]['legs'][0]['duration']['text']
            # results[mode] =f"{origin_direction} to {destination} is {distance} and its time is {duration}"
            response_dict={
                "Mode":f"Mode :{mode.title()}",
                "Distance":f"Distance between {origin_direction} to {destination} is {distance}",
                "Duration":f" Time Duration is {duration}" 
            }
            # return response_dict
        else:
            response_dict={
                "Mode":f"Mode :{mode.title()}",
                "Not Found": f"There is no such route for {mode.title()} for these places."
            }
            
        return response_dict
    
    # function for get the location distance and time between locations
    def find_distance_time_locations(self, dictionary_list, split_user_query):
        location_tag = "Location"
        modes = ['driving', 'walking', 'transit']  
        loc_values = []
        mode_results = []
        for data_dict in dictionary_list:
            values = data_dict.get(location_tag)
            if values is not None and values.lower() in split_user_query:
                if values not in loc_values:
                    loc_values.append(values)
        
        if len(loc_values) == 2:
            print("If condition is running")
            for mode in modes:
                distance_time = self.Calcuate_distance_time(loc_values[0], loc_values[1], mode)
                mode_results.append(distance_time)
            print("1st=====>>",mode_results)
        elif len(loc_values) > 2:
            print("Elif ccondition is running")
            for i in range(len(loc_values) - 1):
                for mode in modes:
                    values = f"{loc_values[i]} {loc_values[i + 1]}".split(" ")
                    distance_time = self.Calcuate_distance_time(values[0], values[1], mode)
                    mode_results.append(distance_time)
        print("mode_results----------->>>",mode_results)

        return mode_results

 
    # def generate_itinerary(data):
    def generate_response(self, data, idx,start=1):
        response_template = Template("""                                                  
        <ul>
      
            {% for key, value in data.items() %}
                {% if key == "Vendor" or key=="Mode" %}
                    <h4>{{ idx + start }}: {{ value }}</h4>
                {% elif key !="Vendor" and  value is not none %}
                    <li> - {{ key }}: {{ value }}</li>
                {% endif %}
            {% endfor %}
        </ul>
        """)
        response = response_template.render(data=data, idx=idx,start=start)
        return response
    
    # make a custom format by using this function
    def get_html_tag(self,ans):
        details = []
        soup = BeautifulSoup(ans, 'html.parser')
        title_element = soup.find('h4') 
        
        for li in soup.find_all('li'):
            key = li.contents[0].strip()
            value = li.contents[1].strip() if len(li.contents) > 1 else ''
            details.append(f'- {key}: {value}')
        itenary_answer = "{}\n{}\n\n".format(title_element, '\n'.join(details)) if title_element else '\n'.join(details) + '\n\n'
        return itenary_answer
    
    
    def post(self , request, format=None):
        answer_found=False
        unique_results = set()         # list of unique header name
        actual_list = []              # use for dictionary list
        dictionary_list = []           ### Store per rows with its header and value in list [{}]
        AnswerImputList=[]
        VandorNameList = []    
        label = ''
        
        itenary_answer=None
        #  Get Data From User "=========================================="
        usr_query=request.data.get("query")                         # input from postman
        topic_id=request.data.get("topic_id")
        vendor_get=request.data.get("vendor_name")
            
        # Preprocess User Query =========================================
        if usr_query:
            correct_user_input=spell(usr_query)
            clean_usr_query=self.clean_text(usr_query)                  # clean and preprocess user query
            split_user_query=clean_usr_query.split(" ")                 # split user query by space
        else: 
            return Response({"Answer":"Data Not Found" },status=status.HTTP_400_BAD_REQUEST) 
        
        if not Topics.objects.filter(user_id=request.user.id).exists():
            data = Topics.objects.create(user_id=request.user.id, name=usr_query)
            topic_id = data.id
        else:
            if not topic_id:
                return Response({'status':status.HTTP_404_NOT_FOUND, 'message':"Please enter topic_id"})
            if not Topics.objects.filter(id=topic_id).exists():
                return Response({'status':status.HTTP_404_NOT_FOUND, 'message':"Invalid topic_id"})
          
        #  Make Greeting Response ====================================== 
        value_found=details_dict.get(correct_user_input.lower().strip())     #  Get greeting values based on the user greeting query
        if value_found:
            itenary_answer=value_found
            answer_found = True 
            
        # Make Response From Database based on user query   =========================================    
        else:
            
            # Dictionary for  change keys name
            replacement_mapping = {
            'net Cost by Experience': 'Cost of Experience',
            'net Cost Per Person Adult': 'Cost Per Person',
            'Maximum Pax per cost': 'Maximum Cost',
            'net Cost Per Person Child Senior': 'Cost Per Person Child/Senior',
            'net Cost by Hour': 'Hourly Cost',
            'Is The Guide Included in the cost': 'Guide Included cost'
            }
            
            # list of keys which is use to adding euro sign
            euro_keys = ['Cost of Experience', 'Cost Per Person Child/Senior', 'Cost Per Person','Maximum Cost']
            
            # list of tag cilumns which is replace by "Associated place"
            tag_list=["tag 1","tag 2","tag 3","tag 4","tag 5","tag 6"]
            
            answer_found = False 
            SelectedVendorData = None
            itenary_answer = None 
            
            # Get data from database
            service = TravelBotData.objects.all().order_by('id')
            for i in service:
                actual_dict = {}
                newList = [i.Vendor, i.net_Cost_by_Experience, i.net_Cost_by_Hour, i.net_Cost_Per_Person_Adult, i.net_Cost_Per_Person_Child_Senior, i.Is_The_Guide_Included_in_the_cost, i.Maximum_Pax_per_cost, i.Location, i.Description_of_the_Experience, i.Time_of_Visit_hours, i.Contact_First_Name, i.Contact_Last_Name, i.Contact_Number, i.Contact_Email, i.Tag_1, i.Tag_2, i.Tag_3, i.Tag_4, i.Tag_5, i.Tag_6]
                
                # ALL columns name get from database
                all_fields = TravelBotData._meta.get_fields()
                field_names = [field.name for field in all_fields]
                del field_names[0]
                
                # Make a List of Dicitinaries of Database data as per row =============================
                count = 0
                for keys in field_names:
                    keys_replace = keys.replace("_", " ")
                    if newList[count] != "nan" and newList[count] != " ":
                        actual_dict[keys_replace] = newList[count]
                        actual_list.append(keys_replace)
                    count += 1
                dictionary_list.append(actual_dict)
        
            # Get list of all header name
            actual_keys = list(set(actual_list))
            # get values of header name and its value frpm the user query
            for items in split_user_query:
                for key in actual_keys:
                    assert items
                    if items in key.lower():
                        unique_results.add(key)
       
            # Call Vendor Function to get Vendor name from user query ========================================
            valuesOfHeader=self.find_vendor_values(dictionary_list , split_user_query)
            if valuesOfHeader:
                AnswerImputList.extend(valuesOfHeader)
                
            # Call Tag Functions to get values from tags column and location columns =======================================
            tagsValues=self.find_tags_values(dictionary_list,split_user_query)
            if tagsValues:
                # print("tagsValues========>>>",tagsValues)
                AnswerImputList.extend(tagsValues)
              
            # Find the result pf google maps
            Google_intent_find=self.findMapIntent(split_user_query)
           
            # Call function for  check user query intent 
            headerToValues=self.get_header_name(actual_keys,split_user_query)
            "-------------------------------------------------------------Final Answer list of dictionary-----------------------------------------"
            filtered_list = [item for item in AnswerImputList if isinstance(item, dict)]
           
            # Make a Code for get result in html tags =================================================================
            if filtered_list and headerToValues:
                print("Both list True")
                result_list = []
                vendor_value="Vendor"
                for index, res_dict in enumerate(filtered_list, start=1):
                    current_dict = {}
                    for Header in headerToValues:
                        if Header.lower() in tag_list:
                            new_header = "Associated Places"
                            new_header = replacement_mapping.get(Header, Header)
                        else:
                            # new_header = Header
                            new_header = replacement_mapping.get(Header, Header)
                        values = res_dict.get(Header)
                        if new_header in euro_keys and values is not None:
                            values = f'€{values}'
                        if values:
                            current_dict["Vendor"]=res_dict[vendor_value] 
                            current_dict[new_header] = values
                    if current_dict:
                        # print('current_dict ===========>>>',current_dict)
                        result_list.append(current_dict.copy())
                    if res_dict[vendor_value] not in  VandorNameList:
                        VandorNameList.append(res_dict[vendor_value])
                # Accumulate responses for each data in result_list
                itenary_answer = [self.generate_response(data,index1) for index1,data in enumerate(result_list)]

            # Condition for when user ask about values of data not column.
            elif len(filtered_list) > 0 and len(headerToValues) == 0:
                print("only filtered_list list True")
                itenary_answer = []
                vendor_value="Vendor"
                for idx2,data in enumerate(filtered_list):
                    updated_data = {}
                    for key, value in data.items():
                        if key.lower() in tag_list:
                            updated_data['Associated Places'] = value
                        else:
                            new_header = replacement_mapping.get(key, key)
                            if new_header in euro_keys and value is not None:
                                value = f'€{value}'  # Corrected this line
                            if value:
                                updated_data[new_header] = value
                            
                    if updated_data[vendor_value] not in VandorNameList:
                        VandorNameList.append(updated_data[vendor_value])
                    print("itenary_answer================>>>",itenary_answer) 
                    itenary_answer.append(self.generate_response(updated_data, idx2))
                    
            # if Vendor list empty then run this code.
            if not VandorNameList:
                print("vendor send from frontend")
                new_dict = {}
                itenary_answer = []  
                vendor_text="Vendor"    
                if vendor_get:
                    vendor_select = ast.literal_eval(vendor_get)
                    for vendor in vendor_select:
                        current_dict1 = {}
                        SelectedVendorData = TravelBotData.objects.filter(Vendor=vendor).values()[0]
                        del SelectedVendorData['id']
                        for original_key, value in SelectedVendorData.items():
                            new_key = original_key.replace("_", " ")
                            new_dict[new_key] = value
                        if new_dict[vendor_text] not in  VandorNameList:
                            VandorNameList.append(new_dict[vendor_text])
                            
                        if headerToValues:
                            for idx3,Header in enumerate(headerToValues):
                                if Header.lower() in tag_list:
                                    new_header = "Associated Places"
                                    new_header = replacement_mapping.get(Header, Header)
                                else:
                                    new_header = Header
                                    new_header = replacement_mapping.get(Header, Header)
                                values = new_dict.get(Header)
                                if new_header in euro_keys and values is not None:
                                    values = f'€{values}'
                                if values !="€nan":
                                    current_dict1['Vendor'] = new_dict[vendor_text]
                                    current_dict1[new_header] = values
                            itenary_answer.append(self.generate_response(current_dict1,idx3))
                        else:
                            return Response({"Answer":"Data Not Found"},status=status.HTTP_400_BAD_REQUEST)
            else:
                itenary_answer
                # Use code for Google maps
                if Google_intent_find:
                    map_rslt_list=self.find_distance_time_locations(dictionary_list , split_user_query)
                    if map_rslt_list:
                        itenary_answer1= [self.generate_response(d_dict, idx4) for idx4, d_dict in enumerate(map_rslt_list)]
                        itenary_answer=itenary_answer+itenary_answer1
                    else:
                        return itenary_answer
                else:
                    itenary_answer
                    
            for ans in itenary_answer:
                extractor.load_document(input=ans, language='en')
                extractor.candidate_selection()
                extractor.candidate_weighting()
                keyphrases = extractor.get_n_best(n=10)
                if keyphrases:  
                    label = keyphrases[0][0]
                answer_found=True
                
            # Make a formatted data by usig this function
            itenary_answer=[self.get_html_tag(ans) for ans in itenary_answer]
            
        # r = requests.post("https://api.deepai.org/api/text-generator",
        #                   {"text":f"Provide  answer from provide context and data in structured form and not print more text only give data values ,{itenary_answer}"}
        #                   ,headers={'api-key':DEEP_API_KEY})
        # genratedText = r.json()
        # itenary=genratedText['output']
        # print("itenary_answer=====>>",itenary)
        if answer_found:
            conversation=UserActivity.objects.create(user_id=request.user.id,questions=usr_query,answer=itenary_answer, topic=label, topic_id_id=topic_id)
            date_time = conversation.date
            datetime_obj = datetime.strptime(str(date_time), "%Y-%m-%d %H:%M:%S.%f%z")  # Use the correct format
            formatted_time = datetime_obj.strftime("%H:%M:%S")
            if Topics.objects.filter(id=topic_id):
                update_Vendor = Topics.objects.filter(id=topic_id).update(vendor_name=VandorNameList)   
            return Response({"Answer":itenary_answer,"time":formatted_time, "id":conversation.id,'label':conversation.topic,"vendor_name":VandorNameList},status=status.HTTP_200_OK)
        else:
            print('answer not found')
            conversation=UserActivity.objects.create(user_id=request.user.id,questions=usr_query,answer="Data not found !! I am in learning Stage.", topic=label, topic_id_id=topic_id)
            return Response({"Answer":"Data not found !! I am in learning Stage."},status=status.HTTP_400_BAD_REQUEST)
                


# API for add info of clients from the form
class UserInfoGethring(APIView):
    authentication_classes=[JWTAuthentication]
    def get(self, request):
        if request.user.id:
            Traveller_Data = UserDetailGethringForm.objects.all().order_by('id').values()
            return Response({"data": Traveller_Data} ,status= status.HTTP_200_OK)
        else:
            return Response({"error":{'message': "Not found!"}},status=status.HTTP_400_BAD_REQUEST)
    def post(self, request):
        EmaployeeName=request.data.get("EmaployeeName")
        TourNumber=request.data.get("TourNumber")
        ClienFirstName=request.data.get("ClienFirstName")
        ClienLastName=request.data.get("ClienLastName")
        Nationalities=request.data.get("Nationalities")
        DatesOfTravel=request.data.get("DatesOfTravel")
        NumberOfTravellers=request.data.get("NumberOfTravellers")
        AgesOfTravellers=request.data.get("AgesOfTravellers")
        BudgetSelect=request.data.get("BudgetSelect")
        FlightArrivalTime=request.data.get("FlightArrivalTime")
        FlightArrivalNumber=request.data.get("FlightArrivalNumber")
        FlightDepartureTime=request.data.get("FlightDepartureTime")
        FlightDepartureNumber=request.data.get("FlightDepartureNumber")
        AccommodationSpecific=request.data.get("AccommodationSpecific")
        MaltaExperience=request.data.get("MaltaExperience")
        StartTime=request.data.get("StartTime")
        LunchTime=request.data.get("LunchTime")
        DinnerTime=request.data.get("DinnerTime")
        IssuesNPhobias=request.data.get("IssuesPhobias")
        OtherDetails=request.data.get("OtherDetails")
        allValuesGet=[
            EmaployeeName,TourNumber,ClienFirstName,ClienLastName,Nationalities,DatesOfTravel,NumberOfTravellers,AgesOfTravellers,BudgetSelect,AccommodationSpecific,MaltaExperience,StartTime,LunchTime,DinnerTime,IssuesNPhobias
        ]
        errorArray=[
            "Employee Name","Number of Tour","Firstname of Client","Lastname of Client","Nationality","Dates of Travel","Number of Travellers","Ages of Travellers","Budget Select","Accommodation Specific and Tags","Malta Experience","Start Time","Lunch Time","Dinner Time","Issues and Phobias"
        ]
        for index,  errorG in enumerate(allValuesGet):
            if errorG=="" or errorG=="null":
                return Response({'error':  {'message': f" The Field {errorArray[index]} is Required !"}},status=status.HTTP_400_BAD_REQUEST)
        if UserDetailGethringForm.objects.filter(numberOfTour=TourNumber).exists():
            return Response({"error":{"message":"This Tour Number is already exist"}},status=status.HTTP_406_NOT_ACCEPTABLE)
        
        elif request.user.id: 
                formsubmit = UserDetailGethringForm.objects.create(
                    user_id=request.user.id,
                    employee_name=EmaployeeName,
                    numberOfTour=TourNumber,
                    client_firstName=ClienFirstName,
                    client_lastName=ClienLastName,
                    nationality=Nationalities,
                    datesOfTravel=DatesOfTravel,
                    numberOfTravellers=NumberOfTravellers,
                    agesOfTravellers=AgesOfTravellers,
                    select_budget=BudgetSelect,
                    flightArrivalTime=datetime.strptime(FlightArrivalTime, "%H:%M").strftime("%I:%M %p"),
                    flightArrivalNumber=FlightArrivalNumber,
                    flightDepartureTime=datetime.strptime(FlightDepartureTime, "%H:%M").strftime("%I:%M %p"),
                    flightDepartureNumber=FlightDepartureNumber,
                    accommodation_specific=AccommodationSpecific,
                    malta_experience=MaltaExperience,
                    start_time= datetime.strptime(StartTime, "%H:%M").strftime("%I:%M %p"),
                    lunch_time=datetime.strptime(LunchTime, "%H:%M").strftime("%I:%M %p"),
                    dinner_time=datetime.strptime(DinnerTime, "%H:%M").strftime("%I:%M %p"),
                    issues_n_phobias=IssuesNPhobias,
                    other_details=OtherDetails
                )
                
                formsubmit.save()
                return Response({'status':{ 'message': "Form Submitted Successfully !"}},status=status.HTTP_201_CREATED,)
        else:
            return Response({'error': { 'message': "User Not Found"}},status=status.HTTP_400_BAD_REQUEST,)
      
# API for make itinerary answer.  

class FRameItinerary(APIView):
    
    # 1 . Function For String Match
    def contains_word(self, s, w):
        return (' ' + w + ' ') in (' ' + s + ' ')
    
   
   
   # 2. Function for get all values from travel data table
    def GetCsvDataFRomDatabase(self,):
        CsvData=[]
        travel_data=TravelBotData.objects.all().values().order_by("id")
        for alldata in travel_data:
            alldata.pop("id", None)                                             # Remove the 'id' key if it exists
            modified_data = {}                                                   # Create a new dictionary for modified data
            count=0
            for key, value in alldata.items():
                if value and  value !="nan":
                    modified_key = key.replace("_", " ")
                    modified_data[modified_key] = value
            CsvData.append(modified_data)
        return CsvData
    
    
    # 3 . Function to Get Dates of Travel on itineray heading (format =  Tue, Dec 05 - Fri, Dec 08 )
    def DatesOfTravel(self,date_string):
        dates = date_string.split(" To ")
        formatted_dates = [datetime.strptime(date, "%a, %b %d, %Y").strftime("%a, %b %d") for date in dates]
        result_string = f"{formatted_dates[0]} - {formatted_dates[1]}"
        return result_string
    
    
    # 4 .function for get total cost for agent and client
    def netANDgross_Value(self , matched_rows,numberOfTravellers):
        Net_value= ['net Cost by Experience','net Cost Per Person Adult','Maximum Pax per cost','net Cost Per Person Child Senior','net Cost by Hour']
        net_values = {}
        gross_value = {}
        for rows_dict in matched_rows:
            vendor = rows_dict.get("Vendor")
            for key, value in rows_dict.items():
                if key in Net_value and value and value != "nan":
                    try:
                        numeric_value = float(value)
                        if vendor not in net_values:
                            net_values[vendor] = 0.0
                        net_values[vendor] += numeric_value
                        
                        if vendor not in gross_value:
                            gross_value[vendor] = 0.0
                        gross_value[vendor] += numeric_value * 1.45
                        
                    except ValueError:
                        print(f"Warning: Non-numeric value '{value}' encountered for key '{key}'")
        net_values_for_agent =round(sum(net_values.values()))* numberOfTravellers
        gross_values_for_agent =round(sum(gross_value.values()))* numberOfTravellers
        return [net_values_for_agent , gross_values_for_agent]
    
    #  5. Function for calculate total days
    def GetDaysFromDate(self, date_string):
        start_date_str, end_date_str = date_string.split(" To ")

        start_date_obj = datetime.strptime(start_date_str, "%a, %b %d, %Y")
        end_date_obj = datetime.strptime(end_date_str, "%a, %b %d, %Y")


        current_date = start_date_obj
        all_days = []
        while current_date <= end_date_obj:
            all_days.append(current_date.strftime("%a, %b %d"))
            current_date += timedelta(days=1)
        response_dict = {
            "Start Day":start_date_obj.strftime("%A"),
            "End Day": end_date_obj.strftime("%A"),
            "All Days": all_days,
            "Total Days": len(all_days),
            # "Month":start_date_obj.strftime("%B")
        }
        return response_dict
    
    # 6 . Google MAp API    (Use in "VendorToVendorTime" function)
    def GetTimeMapAPI(self, origin_direction, destination, mode):
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)
        gps_list = []
        directions_result = gmaps.directions(origin_direction, destination, mode)
        if directions_result:
            distance = directions_result[0]['legs'][0]['distance']['text']
            duration = directions_result[0]['legs'][0]['duration']['text']
            start_location = directions_result[0]['legs'][0]['start_location']
            endlocation = directions_result[0]['legs'][0]['end_location']
            if "day" not in duration:
                gps_list.append((origin_direction, destination,duration, distance,endlocation,start_location))
        return gps_list
    
    # 7. Get Vendor to Vendor Distance   ( Use in post fiunction) 
    def VendorToVendorTime(self, keys_list, Locationslist):
        LocationDistance = []
        error_list=[]
        if len(keys_list) < 2:
            return Response({"error": "Insufficient number of keys"}, status=status.HTTP_400_BAD_REQUEST)
        mapsInputkeys = [Vname.replace(" ", "-").replace("to", "") if Vname else "" for Vname, _ in keys_list]
        for i in range(len(mapsInputkeys) - 1):
            origin = Locationslist[i]
            destination = Locationslist[i + 1]
            key = mapsInputkeys[i]
            formatted_string = f"{key}({origin}) to {mapsInputkeys[i + 1]}({destination})"
            values = formatted_string.split("to")
            # try:
                
            #     distance_time = self.GetTimeMapAPI(values[0], values[1], mode="driving")    
            
            #     if distance_time:
            #         LocationDistance.append(distance_time)
            # except googlemaps.exceptions.ApiError as api_error:
            #     error_list.append({"error": str(api_error), "locations": (values[0], values[1])})
               
        LocationDistance= [[('Hiln-Hotel(Malta) ', ' Verdala-Palace-Private-Visit(Siggiewi)', '19 mins', '12.5 km', {'lat': 35.8617935, 'lng': 14.4009209}, {'lat': 35.9392256, 'lng': 14.3750918})], [('Verdala-Palace-Private-Visit(Siggiewi) ', ' Simon-Cusens-Private-WWII-Collecr(San Gwann)', '21 mins', '14.0 km', {'lat': 35.9079203, 'lng': 14.4782775}, {'lat': 35.8617935, 'lng': 14.4009209})], [('Simon-Cusens-Private-WWII-Collecr(San Gwann) ', ' Simon-Cusens-Private-WWII-Collecr-with-Vehicle-Experience(San Gwann)', '1 min', '1 m', {'lat': 35.9079203, 'lng': 14.4782775}, {'lat': 35.9079203, 'lng': 14.4782775})], [('Simon-Cusens-Private-WWII-Collecr-with-Vehicle-Experience(San Gwann) ', ' Museum-of-Archaeology(Valletta)', '13 mins', '7.7 km', {'lat': 35.8972705, 'lng': 14.5109768}, {'lat': 35.9079203, 'lng': 14.4782775})], [('Museum-of-Archaeology(Valletta) ', ' Museum-of-Archaeology-with-Curar(Valletta)', '1 min', '1 m', {'lat': 35.8972705, 'lng': 14.5109768}, {'lat': 35.8972705, 'lng': 14.5109768})], [('Museum-of-Archaeology-with-Curar(Valletta) ', ' Lascaris-War-Rooms(Valletta)', '7 mins', '1.3 km', {'lat': 35.8950082, 'lng': 14.5126431}, {'lat': 35.8972705, 'lng': 14.5109768})], [('Lascaris-War-Rooms(Valletta) ', ' Notarial-Archives-Private-Visit(Valletta)', '8 mins', '1.8 km', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8950082, 'lng': 14.5126431})], [('Notarial-Archives-Private-Visit(Valletta) ', " St.-John's-Co-Cathedral-Private-Visit-up--10-persons(Valletta)", '10 mins', '2.7 km', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8992325, 'lng': 14.5141017})], [("St.-John's-Co-Cathedral-Private-Visit-up--10-persons(Valletta) ", " St.-John's-Co-Cathedral-Public-Visit(Valletta)", '1 min', '1 m', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8974216, 'lng': 14.5122298})], [("St.-John's-Co-Cathedral-Public-Visit(Valletta) ", " St.-John's-Co-Cathedral-Organ-Charge(Valletta)", '1 min', '1 m', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8974216, 'lng': 14.5122298})], [("St.-John's-Co-Cathedral-Organ-Charge(Valletta) ", ' Fort-St.-Elmo-National-War-Museum(Valletta)', '7 mins', '1.1 km', {'lat': 35.9010413, 'lng': 14.5190351}, {'lat': 35.8974216, 'lng': 14.5122298})], [('Fort-St.-Elmo-National-War-Museum(Valletta) ', ' Muza---National-Museum-of-Art(Valletta)', '4 mins', '0.7 km', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.9010413, 'lng': 14.5190351})], [('Muza---National-Museum-of-Art(Valletta) ', ' Gilder-Visit(Valletta)', '1 min', '1 m', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017})], [('Gilder-Visit(Valletta) ', ' National-Library(Valletta)', '1 min', '1 m', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017})], [('National-Library(Valletta) ', ' Underground-Valletta-Public(Valletta)', '7 mins', '1.4 km', {'lat': 35.8966767, 'lng': 14.5115388}, {'lat': 35.8992325, 'lng': 14.5141017})], [('Underground-Valletta-Public(Valletta) ', ' Underground-Valletta-Book-Out(Valletta)', '1 min', '1 m', {'lat': 35.8966767, 'lng': 14.5115388}, {'lat': 35.8966767, 'lng': 14.5115388})], [('Underground-Valletta-Book-Out(Valletta) ', ' Saluting-Battery-1-Gun-Fire(Valletta)', '7 mins', '1.2 km', {'lat': 35.8950089, 'lng': 14.5126438}, {'lat': 35.8966767, 'lng': 14.5115388})], [('Saluting-Battery-1-Gun-Fire(Valletta) ', ' Saluting-Battery-Full-Gun-Fire(Valletta)', '1 min', '1 m', {'lat': 35.8950089, 'lng': 14.5126438}, {'lat': 35.8950089, 'lng': 14.5126438})], [('Saluting-Battery-Full-Gun-Fire(Valletta) ', ' Dghajsa-Traditional-Water-Taxi-Crossing(Valletta)', '8 mins', '1.8 km', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8950089, 'lng': 14.5126438})], [('Dghajsa-Traditional-Water-Taxi-Crossing(Valletta) ', ' Dghajsa-Traditional-Water-Harbour-Tour(Valletta)', '1 min', '1 m', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017})], [('Dghajsa-Traditional-Water-Harbour-Tour(Valletta) ', ' Stephen-Cordina-Private-Perfume-Workshop(Valletta)', '6 mins', '0.8 km', {'lat': 35.8976456, 'lng': 14.5156008}, {'lat': 35.8992325, 'lng': 14.5141017})], [('Stephen-Cordina-Private-Perfume-Workshop(Valletta) ', ' Stephen-Cordina-Candle-Making-Masterclass(Valletta)', '1 min', '1 m', {'lat': 35.8976456, 'lng': 14.5156008}, {'lat': 35.8976456, 'lng': 14.5156008})], [('Stephen-Cordina-Candle-Making-Masterclass(Valletta) ', ' Upper-Barrakka-Gardens-Lift(Valletta)', '8 mins', '2.0 km', {'lat': 35.8952767, 'lng': 14.5118527}, {'lat': 35.8976456, 'lng': 14.5156008})], [('Upper-Barrakka-Gardens-Lift(Valletta) ', ' Manoel-Theatre-Private-Tour-with-Josette(Valletta)', '7 mins', '2.0 km', {'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.8952767, 'lng': 14.5118527})], [('Manoel-Theatre-Private-Tour-with-Josette(Valletta) ', ' Manoel-Theatre-Public-Entrance(Valletta)', '1 min', '1 m', {'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.8997507, 'lng': 14.5123704})], [('Manoel-Theatre-Public-Entrance(Valletta) ', ' Casa-Rocca-Piccola-Public-Tour(Valletta)', '5 mins', '0.8 km', {'lat': 35.8999431, 'lng': 14.5152405}, {'lat': 35.8997507, 'lng': 14.5123704})], [('Casa-Rocca-Piccola-Public-Tour(Valletta) ', ' Casa-Rocca-Piccola-Private-Afterhours-Tour(Valletta)', '1 min', '1 m', {'lat': 35.8999431, 'lng': 14.5152405}, {'lat': 35.8999431, 'lng': 14.5152405})], [('Casa-Rocca-Piccola-Private-Afterhours-Tour(Valletta) ', ' Boat-Builder-Artisan(Kalkara)', '20 mins', '10.1 km', {'lat': 35.889018, 'lng': 14.5309658}, {'lat': 35.8999431, 'lng': 14.5152405})], [('Boat-Builder-Artisan(Kalkara) ', ' St.-Ursula-Cloistered-Monastery(Valletta)', '21 mins', '9.9 km', {'lat': 35.8975764, 'lng': 14.5154935}, {'lat': 35.889018, 'lng': 14.5309658})], [('St.-Ursula-Cloistered-Monastery(Valletta) ', ' Rolling-Geeks-Experience-with-Lead-Car(Birgu)', '17 mins', '8.4 km', {'lat': 35.888068, 'lng': 14.5221021}, {'lat': 35.8975764, 'lng': 14.5154935})], [('Rolling-Geeks-Experience-with-Lead-Car(Birgu) ', ' Rolling-Geeks-Experience-without-Lead-Car(Birgu)', '1 min', '1 m', {'lat': 35.888068, 'lng': 14.5221021}, {'lat': 35.888068, 'lng': 14.5221021})], [('Rolling-Geeks-Experience-without-Lead-Car(Birgu) ', ' Fort-St.-Angelo-(Birgu)', '6 mins', '1.2 km', {'lat': 35.8918819, 'lng': 14.517297}, {'lat': 35.888068, 'lng': 14.5221021})], [('Fort-St.-Angelo-(Birgu) ', ' Upper-Fort-St.-Angelo-with-a-Knight(Birgu)', '1 min', '1 m', {'lat': 35.8918819, 'lng': 14.517297}, {'lat': 35.8918819, 'lng': 14.517297})], [('Upper-Fort-St.-Angelo-with-a-Knight(Birgu) ', ' Private-Palace-Visit-in-Mdina(Mdina)', '27 mins', '15.2 km', {'lat': 35.8863642, 'lng': 14.4031271}, {'lat': 35.8918819, 'lng': 14.517297})], [('Villa-Bologna-Private-Tour(Attard) ', ' Amy-Scibberas-Art-Resrer-Private-Visit(Qormi)', '8 mins', '4.1 km', {'lat': 35.8785359, 'lng': 14.4705436}, {'lat': 35.8951899, 'lng': 14.4449212})], [('Day-with-the-Chef-in-Malta(None) ', ' Bocci-with-the-Locals(Kalkara)', '33 mins', '22.2 km', {'lat': 35.889018, 'lng': 14.5309658}, {'lat': 35.9392256, 'lng': 14.3750918})], [('Bocci-with-the-Locals(Kalkara) ', ' Sheep-Farm-Visit(Siggiewi)', '23 mins', '12.8 km', {'lat': 35.8537838, 'lng': 14.4385521}, {'lat': 35.889018, 'lng': 14.5309658})], [('Reggatta-Experience(Birgu) ', ' Hastings-Gardens(Valletta)', '21 mins', '10.0 km', {'lat': 35.8976105, 'lng': 14.508432}, {'lat': 35.888068, 'lng': 14.5221021})], [('Museum-of-Natural-Hisry(Mdina) ', " St.-Agatha's-Tower-(Red-Tower)(Mellieha)", '28 mins', '16.9 km', {'lat': 35.9745595, 'lng': 14.342948}, {'lat': 35.8849017, 'lng': 14.403521})], [("St.-Agatha's-Tower-(Red-Tower)(Mellieha) ", ' Sne-Sculpting-Workshop(Valletta)', '46 mins', '27.8 km', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.9745595, 'lng': 14.342948})], [('Sne-Sculpting-Workshop(Valletta) ', " Ta'-Betta-Winery-Private-Tasting(Siggiewi)", '25 mins', '12.6 km', {'lat': 35.8542712, 'lng': 14.4212237}, {'lat': 35.8992325, 'lng': 14.5141017})], [("Ta'-Betta-Winery-Private-Tasting(Siggiewi) ", ' Markus-Divinus-Private-Wine-Tasting(Dingli)', '10 mins', '5.6 km', {'lat': 35.8656269, 'lng': 14.3740761}, {'lat': 35.8542712, 'lng': 14.4212237})], [('Markus-Divinus-Private-Wine-Tasting(Dingli) ', ' The-Citadella(Gozo)', '1 hour 25 mins', '33.7 km', {'lat': 36.0461583, 'lng': 14.2393917}, {'lat': 35.8656269, 'lng': 14.3740761})], [('The-Citadella(Gozo) ', ' Tuk-Tuk-Gozo-Experience-Full-Day(Gozo)', '5 mins', '1.4 km', {'lat': 36.0441118, 'lng': 14.2508575}, {'lat': 36.0461583, 'lng': 14.2393917})]]
        return LocationDistance
    
    
    # Function for updating the location same value
    def update_destination_locations(self , data):
        for nested_list in data:
            destination_mapping = {}
            for entry in nested_list:
                if 'destinationLocation' in entry:
                    destination = entry['destinationLocation']
                    if destination not in destination_mapping:
                        destination_mapping[destination] = entry
                    else:
                        entry['destinationLocation'] = None
        return data
    # 8 . Main Function for Create Dictionary to generate Dictionary    (Use In 'Post' Function)
    def DictOfAllItineraryDAta(self,lead_client_name,datesOfTravel,numberOfTour,net_tripAgent,Gross_tripClient,nationality,AllTripDays,flightArrivalTime,flightArrivalNumber,TourStart_time,lunch_time,perdayresultItinerary,TourTotalDays, malta_experience,get_vehicle_person):
        StartTourKey = False
        Itineary_dict = {}
        Itineary_dict["Lead Client Name"]=lead_client_name
        Itineary_dict["Dates of Travel"]=datesOfTravel
        Itineary_dict["Tour Number"]=numberOfTour
        Itineary_dict["NET Value of trip to the Agent"]=net_tripAgent
        Itineary_dict["Gross Value of the trip to the Client"]=Gross_tripClient
        Itineary_dict["Nationality"]=nationality
        Itineary_dict["Days"]=AllTripDays
        Itineary_dict["Flight Arrival"]=[AllTripDays[0],f"{flightArrivalTime} - Arrival at Malta International Airport on {flightArrivalNumber} and privately transfer to the hotel"]
        Itineary_dict["Vehicle to be used"]=get_vehicle_person
       
        arrival_datetime = datetime.strptime(flightArrivalTime, "%H:%M")
        # Check if the arrival time is before 12:00 PM
        if arrival_datetime.time() < datetime.strptime("12:00", "%H:%M").time():
            StartTourKey = True
        else:
            StartTourKey = False
        
        Itineary_dict["Tour Description"]=[]
            
        if StartTourKey:
                # When Visit time less than 12 vje
            for days in AllTripDays:
                Itineary_dict["Tour Description"].append({days:[]})
        else: 
            # When Visit time above 12 vje
            del AllTripDays[0]
            for days in AllTripDays:
                Itineary_dict["Tour Description"].append({days:[]})
        numberofHours = int(re.search(r'\d+', malta_experience).group())    
        tourDescriptionvalues=Itineary_dict.get('Tour Description')
        totalDays=len(tourDescriptionvalues)
        
        tour_start_datetime = datetime.strptime(TourStart_time, "%H:%M")
        startTime = tour_start_datetime.strftime("%H:%M:%S")
        PerDayActivity=self.DivideDataPerDAyTour(perdayresultItinerary,totalDays,numberofHours,startTime)      # divide list per days itinerary list
        # print("PerDayActivity===============>",len(PerDayActivity), PerDayActivity)
        current_value=startTime
        lunchPrefred=datetime.strptime(lunch_time,"%H:%M")
        lunchPrefredObject=lunchPrefred.strftime("%H:%M:%S")
        SortedDepartList=[]
        for subNewLis in PerDayActivity:
            DepartArrivalList=self.MakevendorDict(subNewLis, startTime,lunch_time)
            lunchEntryDictionary=self.LunchEntry(lunch_time , DepartArrivalList)
            SortedDepartList.append(lunchEntryDictionary)
        count = 0
        
        # Sort list Based on the length of lists max length comes first.
        SortedDepartList = sorted(SortedDepartList, key=len, reverse=True)
        SortedDepartList=  self.update_destination_locations(SortedDepartList)      ### function for change same location to none
        for final_data in tourDescriptionvalues:
            for key, val in final_data.items():
                if count <len(SortedDepartList):
                    final_data[key] = SortedDepartList[count]
                    count += 1
                else:
                    final_data[key] = None
        Itineary_dict["AdditionalInfo"]="This itinerary is just a suggestion, and you can modify it based on your preferences and the time you have available. Always check the opening hours of attractions and plan accordingly. Additionally, consider the current travel guidelines and restrictions that may be in place due to unforeseen circumstances, such as the ongoing global situation."
        return Itineary_dict
    
    # function for sort the array 
    def getist(self, data , val):
        unique_data={}
        arr = [(index, arr, idx, item) for index, arr in enumerate(data) for idx, item in enumerate(val) if item in arr[:-1]]
        getarr = [(getdata[2], getdata[1]) for getdata in arr]
        sorted_getarr = sorted(getarr, key=lambda x: x[0])
        if sorted_getarr:
            for index, values in sorted_getarr:
                if index not in unique_data:
                    if values not in unique_data.values():
                        unique_data[index] = values
        result_list = list(unique_data.values())
        return result_list
    
    # 9.  function for divide number of Vendors based on the per day tours  (Use in "DictOfAllItineraryDAta" function)
    def DivideDataPerDAyTour(self,perdayresultItinerary,TourTotalDays,numberofHours,current_value):  
        tour_end_time = datetime.strptime(current_value, '%H:%M:%S') + timedelta(hours=numberofHours-1)- datetime(1900, 1, 1)
        cumulative_time = timedelta()
        current_day_data=[]
        days=[]
        sorted_days=[]
        for index, data in enumerate(perdayresultItinerary):
            tourstart = datetime.strptime(current_value, '%H:%M:%S')
            duration_parts = data[1].split()
            duration_minutes = int(duration_parts[0])
            duration_timedelta = timedelta(minutes=duration_minutes)
            base_datetime = datetime(2023, 12, 8, 0, 0)
            final_datetime = base_datetime + duration_timedelta
            GoogleMApTime = final_datetime.strftime("%H:%M:%S")
            map_distance_timedelta = datetime.strptime(GoogleMApTime, "%H:%M:%S") - datetime(1900, 1, 1, 0, 0)
            
            visitTime = datetime.strptime(data[3]+":00", "%H:%M:%S") 
            sum_time_current_row = (map_distance_timedelta + visitTime - datetime(1900, 1, 1))
            current_datetime = tourstart + cumulative_time
            
            # Convert tour_end_time - timedelta(minutes=15) to datetime object
            tour_end_datetime = tour_end_time - timedelta(minutes=15)
            if current_datetime.strftime("%H:%M:%S") <= self.format_timedelta_to_HHMMSS(tour_end_datetime):
                cumulative_time += sum_time_current_row
                updated_start_time = (tourstart + cumulative_time).strftime("%H:%M:%S")
                if updated_start_time <=self.format_timedelta_to_HHMMSS(tour_end_datetime):
                    if data not in current_day_data:
                        current_day_data.append(data)
                else:
                    cumulative_time = timedelta()
                    days.append(current_day_data)  
                    current_day_data = []
        if current_day_data:
            if current_day_data not in days:
                days.append(current_day_data)
        OptimizeVendorList=[]
        # sort the vendor based on the far distance from hotel.
        # for lstdata in days:
        #     daysCordinates=[f"{lst[4]['lat']},{lst[4]['lng']}" for lst in lstdata]
        #     OptimizeVendor=generate_itinerary(api_key, daysCordinates, num_days=len(daysCordinates))
        #     if isinstance(OptimizeVendor, dict):
        #         OptimizeVendorList.append([OptimizeVendor])
        # optimizeValues = [[{'lat': float(cordinates.split(',')[0]),'lng': float(cordinates.split(',')[1])} for cordinates in optimizeVendor.values()] for optimizeVendorList in OptimizeVendorList for optimizeVendor in optimizeVendorList ]
        
        # Sort the array based on the distance from hotel to another locations.
        # for data in days:
        #     for val in optimizeValues:
        #         response=self.getist(data , val)
        #         if response and len(response) > 1:
        #             sorted_days.append(response)
                    
        sorted_days=[[[' Verdala-Palace-Private-Visit', '19 mins', 'Siggiewi', '00:00', {'lat': 35.8617935, 'lng': 14.4009209}, {'lat': 35.9392256, 'lng': 14.3750918}], [' Simon-Cusens-Private-WWII-Collecr', '21 mins', 'San Gwann', '1:30', {'lat': 35.9079203, 'lng': 14.4782775}, {'lat': 35.8617935, 'lng': 14.4009209}], [' Simon-Cusens-Private-WWII-Collecr-with-Vehicle-Experience', '1 min', 'San Gwann', '1:30', {'lat': 35.9079203, 'lng': 14.4782775}, {'lat': 35.9079203, 'lng': 14.4782775}]], [[' Museum-of-Archaeology-with-Curar', '1 min', 'Valletta', '0:45', {'lat': 35.8972705, 'lng': 14.5109768}, {'lat': 35.8972705, 'lng': 14.5109768}], [' Lascaris-War-Rooms', '7 mins', 'Valletta', '1:00', {'lat': 35.8950082, 'lng': 14.5126431}, {'lat': 35.8972705, 'lng': 14.5109768}], [" St.-John's-Co-Cathedral-Private-Visit-up--10-persons", '10 mins', 'Valletta', '1:00', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8992325, 'lng': 14.5141017}], [' Notarial-Archives-Private-Visit', '8 mins', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8950082, 'lng': 14.5126431}]], [[" St.-John's-Co-Cathedral-Private-Visit-up--10-persons", '10 mins', 'Valletta', '1:00', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8992325, 'lng': 14.5141017}], [' Notarial-Archives-Private-Visit', '8 mins', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8950082, 'lng': 14.5126431}]], [[" St.-John's-Co-Cathedral-Organ-Charge", '1 min', 'Valletta', '1:00', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8974216, 'lng': 14.5122298}], [' Muza---National-Museum-of-Art', '4 mins', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.9010413, 'lng': 14.5190351}]], [[" St.-John's-Co-Cathedral-Organ-Charge", '1 min', 'Valletta', '1:00', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8974216, 'lng': 14.5122298}], [' Fort-St.-Elmo-National-War-Museum', '7 mins', 'Valletta', '0:30', {'lat': 35.9010413, 'lng': 14.5190351}, {'lat': 35.8974216, 'lng': 14.5122298}], [' Muza---National-Museum-of-Art', '4 mins', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.9010413, 'lng': 14.5190351}], [' Gilder-Visit', '1 min', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017}], [' National-Library', '1 min', 'Valletta', '0:45', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017}]], [[' Muza---National-Museum-of-Art', '4 mins', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.9010413, 'lng': 14.5190351}], [' Gilder-Visit', '1 min', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017}]], [[' Dghajsa-Traditional-Water-Taxi-Crossing', '8 mins', 'Valletta', '0:30', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8950089, 'lng': 14.5126438}], [' Dghajsa-Traditional-Water-Harbour-Tour', '1 min', 'Valletta', '0:15', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017}]], [[' Underground-Valletta-Book-Out', '1 min', 'Valletta', '1:00', {'lat': 35.8966767, 'lng': 14.5115388}, {'lat': 35.8966767, 'lng': 14.5115388}], [' Saluting-Battery-1-Gun-Fire', '7 mins', 'Valletta', '1:00', {'lat': 35.8950089, 'lng': 14.5126438}, {'lat': 35.8966767, 'lng': 14.5115388}], [' Dghajsa-Traditional-Water-Taxi-Crossing', '8 mins', 'Valletta', '0:30', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8950089, 'lng': 14.5126438}], [' Saluting-Battery-Full-Gun-Fire', '1 min', 'Valletta', '0:30', {'lat': 35.8950089, 'lng': 14.5126438}, {'lat': 35.8950089, 'lng': 14.5126438}], [' Dghajsa-Traditional-Water-Harbour-Tour', '1 min', 'Valletta', '0:15', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017}], [' Stephen-Cordina-Private-Perfume-Workshop', '6 mins', 'Valletta', '0:45', {'lat': 35.8976456, 'lng': 14.5156008}, {'lat': 35.8992325, 'lng': 14.5141017}]], [[' Upper-Barrakka-Gardens-Lift', '8 mins', 'Valletta', '2:30', {'lat': 35.8952767, 'lng': 14.5118527}, {'lat': 35.8976456, 'lng': 14.5156008}], [' Manoel-Theatre-Private-Tour-with-Josette', '7 mins', 'Valletta', '00:30', {'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.8952767, 'lng': 14.5118527}], [' Casa-Rocca-Piccola-Public-Tour', '5 mins', 'Valletta', '0:30', {'lat': 35.8999431, 'lng': 14.5152405}, {'lat': 35.8997507, 'lng': 14.5123704}], [' Manoel-Theatre-Public-Entrance', '1 min', 'Valletta', '0:40', {'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.8997507, 'lng': 14.5123704}]], [[' Boat-Builder-Artisan', '20 mins', 'Kalkara', '1:00', {'lat': 35.889018, 'lng': 14.5309658}, {'lat': 35.8999431, 'lng': 14.5152405}], [' St.-Ursula-Cloistered-Monastery', '21 mins', 'Valletta', '0:30', {'lat': 35.8975764, 'lng': 14.5154935}, {'lat': 35.889018, 'lng': 14.5309658}], [' Rolling-Geeks-Experience-with-Lead-Car', '17 mins', 'Birgu', '0:45', {'lat': 35.888068, 'lng': 14.5221021}, {'lat': 35.8975764, 'lng': 14.5154935}]], [[' Amy-Scibberas-Art-Resrer-Private-Visit', '8 mins', 'Mdina', '1:00', {'lat': 35.8785359, 'lng': 14.4705436}, {'lat': 35.8951899, 'lng': 14.4449212}], [' Bocci-with-the-Locals', '33 mins', 'Attard', '0:45', {'lat': 35.889018, 'lng': 14.5309658}, {'lat': 35.9392256, 'lng': 14.3750918}], [' Sheep-Farm-Visit', '23 mins', 'Qormi', '1:00', {'lat': 35.8537838, 'lng': 14.4385521}, {'lat': 35.889018, 'lng': 14.5309658}]], [[" Ta'-Betta-Winery-Private-Tasting", '25 mins', 'Siggiewi', '1:00', {'lat': 35.8542712, 'lng': 14.4212237}, {'lat': 35.8992325, 'lng': 14.5141017}], [' The-Citadella', '1 hour 25 mins', None, '0:30', {'lat': 36.0461583, 'lng': 14.2393917}, {'lat': 35.8656269, 'lng': 14.3740761}], [' Markus-Divinus-Private-Wine-Tasting', '10 mins', 'Rabat', '0:45', {'lat': 35.8656269, 'lng': 14.3740761}, {'lat': 35.8542712, 'lng': 14.4212237}], [' Tuk-Tuk-Gozo-Experience-Full-Day', '5 mins', 'Birgu', '00:30', {'lat': 36.0441118, 'lng': 14.2508575}, {'lat': 36.0461583, 'lng': 14.2393917}]]]
        return sorted_days  
         
    # Function for ading two times  
    def addTime(self,departTime,travelTime):
        if departTime and travelTime:
            DepartTime=datetime.strptime(departTime, "%H:%M:%S")
            TravelTime=datetime.strptime(travelTime, "%H:%M:%S")
            SumOfDepartTravel=DepartTime-datetime(1900,1,1)+TravelTime-datetime(1900,1,1)
            SumOfDepartTravel=self.format_timedelta_to_HHMMSS(SumOfDepartTravel)
            return SumOfDepartTravel
        
    # 10.function for add one hour
    def add_one_hour_after_lunch(self,data_list):
        found_lunch = False
        for idx, data in enumerate(data_list):
            if found_lunch:
                    if 'arrivalTime' in data:
                        data['arrivalTime'] = (datetime.strptime(data['arrivalTime'], '%H:%M:%S') + timedelta(hours=1, minutes=30)).strftime('%H:%M:%S')
                    if "departTime" in data:
                        data['departTime'] = (datetime.strptime(data['departTime'], '%H:%M:%S') + timedelta(hours=1, minutes=30)).strftime('%H:%M:%S')
            if data.get('Vendor') == 'Lunch':
                found_lunch = True
        return data_list

      
   # 11. Function for Chnage timedelta to string
    def format_timedelta_to_HHMMSS(self, td):
        td_in_seconds = td.total_seconds()
        hours, remainder = divmod(td_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        if minutes < 10:
            minutes = "0{}".format(minutes)
        if seconds < 10:
            seconds = "0{}".format(seconds)
        return "{}:{}:{}".format(hours, minutes,seconds)
    
    # 12.function for enter the lunch
    def LunchEntry(self,lunch_time , DepartArrivalList):
        lunchVendor="Lunch"
        lunchTimeObjec = datetime.strptime(lunch_time, '%H:%M')
        lunch=lunchTimeObjec.strftime("%H:%M:%S")
    
        DepartLunchPlus20mintDelta=lunchTimeObjec-datetime(1900,1,1)+timedelta(minutes=20)
        DepartLunch40mintDELTA =lunchTimeObjec-datetime(1900,1,1)-timedelta(minutes=20)               # get 30 minutes time before lunch time
        DepartLunch20mintDELTA =lunchTimeObjec-datetime(1900,1,1)-timedelta(minutes=40)                 # get 1 hour time before lunch time
        
        DepartLunchPlus20mintSTR =self.format_timedelta_to_HHMMSS(DepartLunchPlus20mintDelta)
        DepartLunch40mintSTR =self.format_timedelta_to_HHMMSS(DepartLunch40mintDELTA)
        DepartLunch20mintSTR =self.format_timedelta_to_HHMMSS(DepartLunch20mintDELTA)
        
        # Add Lunch in Departarrive list when selected 6 and 8 hour tour per day
        for idx,entry in enumerate(DepartArrivalList):
            ArriveTime=entry.get("arrivalTime")
            VisitTime=entry.get("visitTime")
            if ArriveTime is not None and VisitTime is not None:
                SumOfArriveAndVisitTime=self.addTime(ArriveTime ,VisitTime)
                # check if SumOfArriveAndVisitTime exist between 20 mint up and down from lunch time
                if lunch <= SumOfArriveAndVisitTime <= DepartLunchPlus20mintSTR:  
                    print('if condtion is running')
                    lunchtime =lunchTimeObjec-datetime(1900,1,1) +timedelta(minutes=20)
                    if all(data.get("Vendor") != lunchVendor for data in DepartArrivalList):
                        DepartArrivalList.insert(idx+1 ,{"departTime": SumOfArriveAndVisitTime,"destinationLocation":"Restaurant"})
                        DepartArrivalList.insert(idx+2, {"arrivalTime": self.format_timedelta_to_HHMMSS(lunchtime), "Vendor": "Lunch", "arrivedVendor": "Lunch Break"})
                    break
                elif DepartLunch20mintSTR <= SumOfArriveAndVisitTime <= DepartLunch40mintSTR:
                    print('elif condtion  1 is running')
                    lunchtime2 =lunchTimeObjec-datetime(1900,1,1) - timedelta(minutes=15)
                    if all(data.get("Vendor") != lunchVendor for data in DepartArrivalList):
                        DepartArrivalList.insert(idx+1,{"departTime": SumOfArriveAndVisitTime,"destinationLocation":"Restaurant"})
                        DepartArrivalList.insert(idx+2, {"arrivalTime": self.format_timedelta_to_HHMMSS(lunchtime2), "Vendor": "Lunch", "arrivedVendor": "Lunch Break"})
                    break
                elif DepartLunch40mintSTR <= SumOfArriveAndVisitTime <= lunch:
                    print('elif condtion  2 is running')
                    if all(data.get("Vendor") != lunchVendor for data in DepartArrivalList):
                        DepartArrivalList.insert(idx+1,{"departTime": SumOfArriveAndVisitTime,"destinationLocation":"Restaurant"})
                        DepartArrivalList.insert(idx+2, {"arrivalTime": lunch, "Vendor": "Lunch", "arrivedVendor": "Lunch Break"})
                    break
                
       # add lunch when cleint select 4 hour tour per day.
        if all(data.get("Vendor") != lunchVendor for data in DepartArrivalList):
            getlastRow = DepartArrivalList[-1]
            SumOfDepartTravel = self.addTime(getlastRow.get("arrivalTime"), getlastRow.get("visitTime"))
            if SumOfDepartTravel <= lunch:
                DepartArrivalList.append({"arrivalTime": lunch, "Vendor": "Lunch", "arrivedVendor": "Lunch Break"})
       
        #get first row from  the lunch row
        lunch_index = None
        previous_row = None
        for idx , row in enumerate(DepartArrivalList):
            if row.get("Vendor")==lunchVendor:
                lunch_index=idx
                previous_index = idx - 1 if idx > 0 else None
                if previous_index is not None:
                    previous_row = DepartArrivalList[previous_index]
                break
        if previous_row:
            if "arrivalTime" in previous_row and "visitTime" in previous_row:
                AddTime =self.addTime(previous_row.get("arrivalTime") ,previous_row.get("visitTime"))
                DepartLunch=lunchTimeObjec-datetime(1900,1,1)-timedelta(minutes=10)
                if AddTime <= self.format_timedelta_to_HHMMSS(DepartLunch):
                    DepartArrivalList.insert(lunch_index ,{"departTime":AddTime ,"destinationLocation":"Restaurant"})
        
        # #Add Departure Dictionary for Hotel  
        getlastRoe=DepartArrivalList[-1]
        SumOfDepartTravel =self.addTime(getlastRoe.get("arrivalTime") ,getlastRoe.get("visitTime"))
       
        # If condtion is working for 6 and 8 hour data
        updatedDepartArrivalList=[]
        if SumOfDepartTravel:
            DepartArrivalList.append({"departTime":SumOfDepartTravel ,"destinationLocation":"Hotel"})
            updatedDepartArrivalList=self.add_one_hour_after_lunch(DepartArrivalList)
            
        # else condition working for 4 hour data
        else:
            lunchshift30mint =lunchTimeObjec-datetime(1900,1,1)-timedelta(minutes=30)               # get 30 minutes time before lunch time
            lunchshiftonehour =lunchTimeObjec-datetime(1900,1,1)-timedelta(hours=1)                 # get 1 hour time before lunch time
            lunchshift30mint =self.format_timedelta_to_HHMMSS(lunchshift30mint)
            lunchshiftonehour =self.format_timedelta_to_HHMMSS(lunchshiftonehour)
            
            # get Second last row of list.
            getSeondLasTRow=DepartArrivalList[-2]
            departTime=getSeondLasTRow.get("departTime")
            
            # if departime varied near lunch then lunch time remain same
            if departTime >= lunchshift30mint:    
                returntime =lunchTimeObjec-datetime(1900,1,1) +timedelta(minutes=50)
                returntime =self.format_timedelta_to_HHMMSS(returntime)
                DepartArrivalList.append({"departTime":returntime ,"destinationLocation":"Hotel"})
                updatedDepartArrivalList=DepartArrivalList

            # if Departaddtimee is varify between lunchshiftonehour and lunchshift30mint then this condition will work
            else:
                
                lunchshift15mint =lunchTimeObjec-datetime(1900,1,1)-timedelta(minutes=15)
                lunchshift15mint =self.format_timedelta_to_HHMMSS(lunchshift15mint)
                returnhotel = datetime.strptime(lunchshift15mint, "%H:%M:%S")
                returnhotel += timedelta(hours=1)
                returnhotel = returnhotel.strftime("%H:%M:%S")
                
                # if lunch time has gap then shift lunch 15 minutes befroe
                if lunchshiftonehour <= departTime <= lunchshift30mint:
                    for data in DepartArrivalList:
                        if data.get("Vendor") == lunchVendor:
                            data['arrivalTime'] = lunchshift15mint
                DepartArrivalList.append({"departTime":returnhotel ,"destinationLocation":"Hotel"})
                updatedDepartArrivalList = DepartArrivalList  
        return updatedDepartArrivalList
    
    # 13. Make Vendor and Depart data list
    def MakevendorDict(self,data , starttime,lunch_time):
        lunchTimeObjec = datetime.strptime(lunch_time, '%H:%M')
        lunch=lunchTimeObjec.strftime("%H:%M:%S")
        Depart_dict2 = {}
        Depart_dict2['departTime']=[]
        # departTime = "10:00:00"
        departTime = starttime
        finalList = []
        index = 0
        arrivedVendor = ""
        for k in range(len(data)):
            duration_parts = data[k][1].split()
            duration_minutes = int(duration_parts[0])
            duration_timedelta = timedelta(minutes=duration_minutes)
            base_datetime = datetime(2023, 12, 8, 0, 0)
            final_datetime = base_datetime + duration_timedelta
            
            GoogleMApTime = final_datetime.strftime("%H:%M:%S")
            map_distance_timedelta = datetime.strptime(GoogleMApTime, "%H:%M:%S") - datetime(1900, 1, 1, 0, 0)
            
            departedVendor = data[k][2]
            visitTime = datetime.strptime(data[k][3]+":00", "%H:%M:%S") 

            if index %2 == 0:
                finalList.append({"departTime":departTime,"travelTime":GoogleMApTime,"Vendor":departedVendor,"destinationLocation":data[k][2]})
                arrivedVendor =  data[k][0]
                departTime=datetime.strptime(departTime, "%H:%M:%S")
                arrivalTime = map_distance_timedelta + departTime  - datetime(1900,1,1)
                arrivalTime = self.format_timedelta_to_HHMMSS(arrivalTime)
                finalList.append({"arrivalTime":arrivalTime,"visitTime":visitTime.strftime("%H:%M:%S"),"arrivedVendor":arrivedVendor,"destinationLocation":data[k][2]})
                departedVendor = data[k][0]
                arrivalTime=datetime.strptime(arrivalTime, "%H:%M:%S")
                departTime = arrivalTime -  datetime(1900,1,1) + visitTime -  datetime(1900,1,1)
                departTime = self.format_timedelta_to_HHMMSS(departTime)
            else:
                finalList.append({"departTime":departTime,"travelTime":GoogleMApTime,"Vendor":departedVendor,"destinationLocation":data[k][2]})
                arrivedVendor =  data[k][0]
                departTime=datetime.strptime(departTime, "%H:%M:%S")
                arrivalTime = map_distance_timedelta + departTime  - datetime(1900,1,1)
                arrivalTime = self.format_timedelta_to_HHMMSS(arrivalTime)
                finalList.append({"arrivalTime":arrivalTime,"visitTime":visitTime.strftime("%H:%M:%S"),"arrivedVendor":arrivedVendor,"destinationLocation":data[k][2]})
                departedVendor = data[k][0]
                arrivalTime=datetime.strptime(arrivalTime, "%H:%M:%S")
                departTime = arrivalTime -  datetime(1900,1,1) + visitTime -  datetime(1900,1,1)
                departTime = self.format_timedelta_to_HHMMSS(departTime)
            index += 1
            
        # print('finallist---------------------->>>>',finalList)
        return finalList
    
    # 14.  Generate Itinerary      (Use in " Post Function")
    def GenerateItineraryResponse(self , FramedItinerary):
        response_template=Template('''
    <div className="tour-discription">
        <div className="discription">
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>Lead Client Name:</b> {{ FramedItinerary.get('Lead Client Name')|trim }}</p>
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>Dates of Travel:</b> {{ FramedItinerary.get('Dates of Travel')|trim }}</p>
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>Tour Number:</b> {{ FramedItinerary.get('Tour Number')|trim }}</p>
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>NET Value of trip to the agent:</b> {{ FramedItinerary.get('NET Value of trip to the Agent')|trim }}</p>
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>Gross Value of the trip to the client:</b> {{ FramedItinerary.get('Gross Value of the trip to the Client')|trim }}</p>
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>Vehicle to be used:</b> {{ FramedItinerary.get('Vehicle to be used')|trim }}</p>
            <p className="d-content" style="font-family:sans-serif; font-size:18px"><b>Nationality:</b> {{ FramedItinerary.get('Nationality')|trim }}</p>
            <br>
            <div className="data-per-day">
              <ul className="data">     
                {% for key, value in FramedItinerary.items() %}
                    {% if key == "Flight Arrival" %}
                        <h5 className="Date-format" style="font-size: 20px; font-weight: 500; line-height: 50px">{{ value[0]|trim }}:</h5> 
                        <p className="d-content" style="font-family:sans-serif; font-size:16px">{{ value[1]|trim }}</p>
                    {% elif key == "Tour Description" %}
                        {% for data in value %}
                            {% for dates, description in data.items() %}
                                <h5 className="Date-format" style="font-size: 20px; font-weight: 500; line-height: 50px">{{ dates|trim }}</h5>
                                {% if description != None %}
                                    {% if dates %}
                                        {% for entry in description %}
                                            {% if "departTime" in entry %}
                                                {% if loop.first %}
                                                    <li className="data-per-date" style="font-size: 16px; font-family: sans-serif; line-height: 30px">{{ entry.departTime }} : {{ "Depart with local guide and driver for " ~ entry.destinationLocation }}</li>
                                                {% else %}
                                                    {% if entry.destinationLocation != None %}
                                                    <li className="data-per-date" style="font-size: 16px; font-family: sans-serif; line-height: 30px">{{ entry.departTime }} : {{ "Depart for " ~ entry.destinationLocation }}</li>
                                                    {% endif %}
                                                {% endif %}
                                            {% elif "arrivalTime" in entry %}
                                                    <li className="data-per-date" style="font-size: 16px; font-family: sans-serif; line-height: 30px">{{ entry.arrivalTime }} : {{ entry.arrivedVendor }}</li>
                                            {% endif %}
                                        {% endfor %}
                                    {% endif %}
                                {% elif description == None %}
                                    <p className="d-content" style="font-family:sans-serif, font-size:16px">{{ description }}</p>
                                {% endif %}
                            {% endfor %}
                        {% endfor %}
                    {% endif %}
                {% endfor %}
              </ul>
            </div>
        </div>
        <h4 style ="font-size: 20px; font-weight: 500; line-height: 50px, font-family:sans-serif"><b>Note :</b> </h4>
        <p className="d-content" style="font-family:sans-serif; font-size:18px">{{ FramedItinerary.get('AdditionalInfo')|trim  }}</p> 
    </div>

        ''')
        response = response_template.render(FramedItinerary=FramedItinerary)
        print("response=========>>",response)
        return response

        # function for provide vehicle to person
    
    # 15 Vehicle Provide Function
    def VehicleProvide(self,person_check):
        vehicle1="Mercedes E Class"
        vehicle2="Mercedes Vito"
        if person_check:
            if person_check < 3:
                return vehicle1
            elif person_check >=3 and person_check <= 6:
                return vehicle2
        else:
            return Response({'error':'Data Not Found'},status=status.HTTP_404_NOT_FOUND)       
    
    # not go falconry in july,aug and sep
    def MonthlyRestricition(self,month):
        month_name ="July"
        NotIncludeMonth=["July","Aug","Sep"]
        if month_name in NotIncludeMonth:
            return "Falconry is Not Avaliable"
        else:
            return "None"
   
    def post(self,request , format=None):
        rules_keys=["Days","Valletta","Marsaxlokk"," Net Values","driving distance","Falconry(july, August,Sep)"]
        map_obj=Prediction()

        form_id=request.data.get("form_id")
       
        if not form_id :
            return Response({"message":"Form ID is required"},status=status.HTTP_204_NO_CONTENT)
        
        if not UserDetailGethringForm.objects.filter(id=form_id).exists():
            return Response({"error": "Form ID not Exist"}, status=status.HTTP_404_NOT_FOUND)
        else:
            # get Form data from Database.
            query_result = UserDetailGethringForm.objects.filter(id=form_id).values().order_by("id")
            
            # get all data from 
            AllCsvData=self.GetCsvDataFRomDatabase()
            
            itinerary_dict={}  
            
            for form_data in query_result:
                lead_client_name=form_data.get("employee_name")                     # LEad client name
                date_striing=form_data.get('datesOfTravel')                         # Date of Travel
                datesOfTravel=self.DatesOfTravel(date_striing)                      # only for heading tag
                numberOfTour=form_data.get('numberOfTour')                          # Tour NUmber
                nationality=form_data.get("nationality")
                numberOfTravellers=form_data.get('numberOfTravellers')            
                TourStart_time = datetime.strptime(form_data.get("start_time"), "%I:%M %p").strftime("%H:%M")
                lunch_time = datetime.strptime(form_data.get("lunch_time"), "%I:%M %p").strftime("%H:%M")
                dinner_time = datetime.strptime(form_data.get("dinner_time") , "%I:%M %p").strftime("%H:%M")
                flightArrivalTime = datetime.strptime(form_data.get("flightArrivalTime"), "%I:%M %p").strftime("%H:%M")
                malta_experience=form_data.get("malta_experience")                  # malta_experience
                flightArrivalNumber=form_data.get("flightArrivalNumber")

                get_vehicle_person=self.VehicleProvide(numberOfTravellers)          # Provide vehicle based on the person.
              
                # code for get row based on the tag
                tag_accomodation_value=form_data.get("accommodation_specific").lower()        
                split_accomodation=re.split("[&/,]| or",tag_accomodation_value)
                
                "----------------------------------------------------------------------------------------------"
                # Get data Row Based on the form Tag.
                matched_rows=map_obj.find_tags_values(AllCsvData,split_accomodation)             
                # print("matched_rows==========================>>",len(matched_rows),matched_rows)
                netAnd_GROSS=self.netANDgross_Value(matched_rows,numberOfTravellers)            # Get value trip value for client and person
                
                itinerary_dict["lead_client_name"]=lead_client_name
                itinerary_dict["datesOfTravel"]=datesOfTravel  
                itinerary_dict["numberOfTour"]=numberOfTour
                itinerary_dict["numberOfTravellers"]=numberOfTravellers
                itinerary_dict["Net Trip Value Agent"]=f"€{netAnd_GROSS[0]}"      
                itinerary_dict["Gross Trip Value Client"]=f"€{netAnd_GROSS[1]}"
              
                
                # # 1. find Total days , date , month 
                total_days_dict=self.GetDaysFromDate(date_striing)
                itinerary_dict.update(total_days_dict)

            # get net and gross value
            net_tripAgent=itinerary_dict.get("Net Trip Value Agent")
            Gross_tripClient=itinerary_dict.get("Gross Trip Value Client")
            TourTotalDays=itinerary_dict.get("Total Days")
            # print("matched rows",matched_rows, len(matched_rows))
            # get list of all trip days
            AllTripDays=itinerary_dict.get("All Days") 

            StartingPointList=["Valletta","Senglea", "St Julians"] 
            VendorTimeVistHour={}
            Locationslist=["Malta"]
            # Locationslist=[]
            VendorTimeVistHour["Hilton Hotel","00:00"]="00:00"
          
            for data_dict in matched_rows:
                Vendors=data_dict.get("Vendor")  
                time=data_dict.get("Time of Visit hours")      
                Locationslist.append(data_dict.get("Location") )
                VendorTimeVistHour[Vendors, time]=time
            # get distance time between places     
            keys_list=list(VendorTimeVistHour.keys())
            ActivityVisitTime=(VendorTimeVistHour.values())
            TimeLoc=self.VendorToVendorTime(keys_list,Locationslist)
            # add time , location and vendor in this list.
            perdayresultItinerary=[]        
            del Locationslist[0]
            for sublist, location,timeVist in zip(TimeLoc, Locationslist,ActivityVisitTime):
                if str(timeVist) == "None" or timeVist.strip() == "" or timeVist == "0:00":
                    timeVist = "00:30"
                for j in sublist:
                    perdayresultItinerary.append([j[1].split("(")[0], j[2], location,timeVist,j[4],j[5]])
            Gotitinerary_dict=self.DictOfAllItineraryDAta(lead_client_name,datesOfTravel,numberOfTour,net_tripAgent,Gross_tripClient,nationality,
                        AllTripDays,flightArrivalTime,flightArrivalNumber,TourStart_time,lunch_time,perdayresultItinerary,TourTotalDays, malta_experience,get_vehicle_person)
            Framed_response=self.GenerateItineraryResponse(Gotitinerary_dict)
            if Framed_response:
                return Response({'data': Framed_response , "message":"success"},status=status.HTTP_200_OK)
            else:
                return Response({"message":"Data Not Found"},status=status.HTTP_404_NOT_FOUND)
      
class GetFormDetails(APIView):
    authentication_classes=[JWTAuthentication]
    def post(self , request ,format=None):
        GetformId=request.data.get("form_id")
        if not GetformId:
            return Response({'Message':'Please Enter Form ID'}, status=status.HTTP_400_BAD_REQUEST,)

        if not UserDetailGethringForm.objects.filter(id=GetformId).exists():
                return Response({'Message':'Form ID not Exist'},status=status.HTTP_400_BAD_REQUEST,)
            
        getdata=UserDetailGethringForm.objects.filter(id=GetformId).all().values()[0]
        if getdata:
            return Response({"Message":"Data Get SuccessFully","data":getdata},status=status.HTTP_200_OK)
        else:
            return Response({"Message":"Data Not Found"},status=status.HTTP_404_NOT_FOUND)

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
            data = Topics.objects.filter(user_id=user.data['id']).values('id','user_id','name' , 'created_at__date','vendor_name')
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