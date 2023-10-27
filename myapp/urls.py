from django.urls import path 
from myapp.views import *
from myapp import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('register/', UserRegistrationView.as_view(),name='register'),
    path('login/', UserLoginView.as_view(),name='loginin'),
    path('userprofile/', UserProfileView.as_view(),name='userprofile'),
    path('logout/', LogoutUser.as_view(),name='logout'),
    path('userlist/', UserList.as_view(),name='csvupload'),
    path('csvupload/', UploadCsv.as_view(),name='csvupload'),
    # path('trainmodel/', TrainModel.as_view(),name='trainmodel'),
    path('prediction/', prediction.as_view(),name='prediction'),
    path('userhistory/', GetUserHistory.as_view(),name='userhistory'),
    path('add-suggestion/', AddQuestionAnswer.as_view(),name='addsuggestion'),
    path('csvfilehistory/', GetAlluploadedcsv.as_view(),name='csvfilehistory'),    
    path('userdelete/<int:id>/', DeleteUser.as_view(),name='userdelete'),
    path('suggestion/<int:id>/', AnswerSuggestion.as_view(),name='answersuggestion'),
    path('profileupdate/<int:id>/', ProfileUpdate.as_view(), name='profileupdate'),
    path('getchat/<int:topic_id>/', ChatDetailsByID.as_view(), name='getchat'),
    path('userstatus/<int:id>/', ActiveInactive.as_view(), name='userstatus'),
    path('changepassword/', UserChangePasswordView.as_view(), name='changepassword'),
    path('deletefile/<int:id>/', CsvDeleteView.as_view(), name='deletefile'),
    path('topics/', TopicsView.as_view(), name='topics'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)