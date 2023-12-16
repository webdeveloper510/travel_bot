from django.test import TestCase

'''

  def DictOfAllItineraryDAta(self,lead_client_name,datesOfTravel,numberOfTour,net_tripAgent,Gross_tripClient,AllTripDays,flightArrivalTime,flightArrivalNumber,StartingPointList,TourStart_time,lunch_time,perdayresultItinerary,TourTotalDays, malta_experience):
        StartTourKey = False
        Itineary_dict = {}
        Itineary_dict["Lead Client Name"]=lead_client_name
        Itineary_dict["Dates of Travel"]=datesOfTravel
        Itineary_dict["Tour Number"]=numberOfTour
        Itineary_dict["NET Value of trip to the Agent"]=net_tripAgent
        Itineary_dict["Gross Value of the trip to the Client"]=Gross_tripClient
        Itineary_dict["Days"]=AllTripDays
        Itineary_dict["Flight Arrival"]=[AllTripDays[0],f"{flightArrivalTime} - Arrival at Malta International Airport on {flightArrivalNumber} and privately transfer to the hotel"]
        
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
                Itineary_dict["Tour Description"].append({days:{}})
        else: 
            # When Visit time above 12 vje
            del AllTripDays[0]
            for days in AllTripDays:
                Itineary_dict["Tour Description"].append({days:{}})
        
        tour_start_datetime = datetime.strptime(TourStart_time, "%H:%M")
        startTime = tour_start_datetime.strftime("%H:%M:%S")
        PerDayActivity=self.DivideDataPerDAyTour(perdayresultItinerary,TourTotalDays)      # divide list per days itinerary list
        current_value=startTime
        TimeValues = []
        VendorValues = []
        LocationValues = []
        numberofHours = int(re.search(r'\d+', malta_experience).group())
        for subNewLis in PerDayActivity:
            num_groups = len(subNewLis)
            for j, day_list in enumerate(subNewLis):
                if j % num_groups == 0:
                    startTime = current_value
                
                # Assuming self.ItineraryTimeValues returns startTime, Vendorname, location
                startTime, Vendorname, location = self.ItineraryTimeValues(startTime, day_list)
                TimeValues.append(startTime)
                VendorValues.append(Vendorname)
                LocationValues.append(location)
        tourDescriptionvalues=Itineary_dict.get('Tour Description')
        totalDays=len(tourDescriptionvalues)
        
        
        PerDaySchduleDict=self.MakeDaySchduleDict(totalDays,numberofHours,TimeValues,VendorValues,LocationValues,current_value,lunch_time)
        # tourDescriptionvalues=Itineary_dict.get('Tour Description')
        count = 0
        for data in tourDescriptionvalues:
            valuesList  = list(PerDaySchduleDict.values())
            for k_ ,v_ in data.items():
                data[k_] = valuesList[count]
            count+=1
        return Itineary_dict
    
'''