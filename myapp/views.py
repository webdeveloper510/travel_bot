
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
import json
import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from autocorrect import Speller
spell=Speller(lang='en')



# url="http://127.0.0.1:8000/static/media/"
url="http://16.171.134.22:8000/static/media/"

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
        users = User.objects.all().order_by('id').values()
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
            expected_columns = ["question", "answer", "label"]
            
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
                questions = row["question"]
                answers = row["answer"]
                label = row['label']
                if not TravelBotData.objects.filter(question=questions , answer=answers).exists():
                    dataload = TravelBotData.objects.create(question=questions, answer=answers, topic_name=label)
                    dataload.save()
            return Response({'message': "File uploaded and data saved successfully"})

        except Exception as e:
            return Response({'status': status.HTTP_500_INTERNAL_SERVER_ERROR, 'message': str(e)})
        
# Api for train model  
class TrainModel(APIView):
    authentication_classes=[JWTAuthentication]
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
    
    def post(self, request):
        traveldata=TravelBotData.objects.all().order_by('id')
        serializer =TravelBotDataSerializer(traveldata, many=True)
        array=[]
        for data in serializer.data:
            questions=self.clean_text(data['question'])
            answer=data['answer']
            topic_name=data['topic_name']
            data_dict={"Topic":topic_name,"question":questions,"answer":answer}
            array.append(data_dict)
        Questions=[dict['question'] for dict in array]
        # # define the parameter for model.
        MAX_NB_WORDS = 1000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        sequence= tokenizer.texts_to_sequences(Questions)
        ## Create input for model
        input_data=pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)             # input 
        
        # convert label into dummies using LabelEncoder.
        Y_data = [dict['Topic'].strip() for dict in array]
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(Y_data)
        output_Y = lbl_encoder.transform(Y_data) 
        cluster_label = lbl_encoder.classes_.tolist()
        num_class=len(cluster_label)
        # define the layers for sequential model.
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        epochs =500
        batch_size=128
        model.fit(input_data, np.array(output_Y), epochs=epochs, batch_size=batch_size)
        # # Save Model
        model_json=model.to_json()
        with open(os.getcwd()+"/saved_model/classification_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.getcwd()+"/saved_model/classification_model_weights.h5")
        # Save the cluster label list
        with open(os.getcwd()+"/saved_model/cluster_labels.pkl", "wb") as file:
            pickle.dump(cluster_label, file)
        return Response({'message':"Model Train Successfully"})


details_dict={
    "hi":"Hello",
    "hey" :"Hi",
    "is anyone there?" :"Hi there ",
    "is anyone there" :"Hi there ",
    "hello" :"Hi",
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
    model_path=os.getcwd()+"/saved_model/classification_model.json"
    model_weight_path=os.getcwd()+"/saved_model/classification_model_weights.h5"
    cluster_label_path=os.getcwd()+'/saved_model/cluster_labels.pkl'
    clean_func=TrainModel()     
    
    def post(self, request):
        service = TravelBotData.objects.all().order_by('id')
        serializer = TravelBotDataSerializer(service, many=True)
        array=[]
        for x in serializer.data:
            question=self.clean_func.clean_text(x["question"])
            answer=x["answer"]
            TopicName=(x['topic_name'])
            data_dict={"Topic":TopicName,"question":question,"answer":answer}
            array.append(data_dict)
            
        Questions=[dict['question'] for dict in array]
        # Tokenizer data.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        
        # get the json labels
        with open(self.cluster_label_path, "rb") as file:
            cluster_labels = pickle.load(file)
        # Load Saved Model.
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.model_weight_path)
        # Take user input and preprocess as input
        user_input=request.data.get("query")
        input=spell(user_input)
        value_found=details_dict.get(input.lower().strip())
        clean_user_input=self.clean_func.clean_text(input)
        new_input = tokenizer.texts_to_sequences([clean_user_input])
        new_input = pad_sequences(new_input, maxlen=MAX_SEQUENCE_LENGTH) 
        
        # Make Prediction
        pred =loaded_model.predict(new_input)
        databasew_match=pred, cluster_labels[np.argmax(pred)]
        result=databasew_match[1]
        # Get the answer based on the question.
        filter_data = [dict for dict in array if dict["Topic"].strip()== result.strip()]
        get_all_questions=[dict['question'] for dict in filter_data] 
        vectorizer = TfidfVectorizer()
        vectorizer.fit(get_all_questions)
        question_vectors = vectorizer.transform(get_all_questions)                                  # 2. all questions
        input_vector = vectorizer.transform([clean_user_input])
        # check the similarity of the model
        similarity_scores = question_vectors.dot(input_vector.T).toarray().flatten()  # Ensure 1-dimensional array
        max_sim_index = np.argmax(similarity_scores)
        similarity_percentage = similarity_scores[max_sim_index] * 100
        if (similarity_percentage)>=65:
            answer = filter_data[max_sim_index]['answer'] 
            conversation=UserActivity.objects.create(user_id=request.user.id,questions=input,answer=answer)
            conversation.save()
            response_data = {
                "Question":input,
                "Label": result,
                "Answer": answer,
            
                "AnswerSource":"This Response Done From Database"}
        elif value_found:
                response_data = {"Question":input,"Answer": value_found}
                return Response({"data":response_data,"code":200})
        else:
            response_data = {"Message":"Data Not Found"}
            
        return Response({"data":response_data,"code":200})
     
# Api for get user history   
class GetUserHistory(APIView):
    authentication_classes=[JWTAuthentication]
    def get(self,request):
        
        history=UserActivity.objects.filter(user_id=request.user.id).values()
        print(history)
        return Response({"data":history,"code":200})
    
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


        
         
            