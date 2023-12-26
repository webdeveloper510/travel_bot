# data = [
#     [['Notarial-Archives-Private-Visit', '6 mins', 'Valletta', '00:30', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.89780349999999, 'lng': 14.5118479}],
#      ["St.-John's-Co-Cathedral-Private-Visit-up--10-persons", '10 mins', 'Valletta', '00:30', {'lat': 35.8974216, 'lng': 14.5122298}, {'lat': 35.8992325, 'lng': 14.5141017}], 
#      ['Muza---National-Museum-of-Art', '5 mins', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8974216, 'lng': 14.5122298}], 
#      ['Gilder-Visit', '1 min', 'Valletta', '1:00', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8992325, 'lng': 14.5141017}], 
#      ['Stephen-Cordina-Private-Perfume-Workshop', '6 mins', 'Valletta', '0:45', {'lat': 35.8976456, 'lng': 14.5156008}, {'lat': 35.8992325, 'lng': 14.5141017}]],
#     [['Manoel-Theatre-Private-Tour-with-Josette', '7 mins', 'Valletta', '2:30', {'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.8976456, 'lng': 14.5156008}], 
#      ['Manoel-Theatre-Public-Entrance', '1 min', 'Valletta', '0:40', {'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.8997507, 'lng': 14.5123704}], 
#      ['Palazzo-Parisio-with-Tour-and-Afternoon-Tea', '22 mins', 'Valletta', '0:30', {'lat': 35.9148366, 'lng': 14.444285}, {'lat': 35.8997507, 'lng': 14.5123704}]],
#     [['Sheep-Farm-Visit', '22 mins', 'Qormi', '1:00', {'lat': 35.8537838, 'lng': 14.4385521}, {'lat': 35.9392256, 'lng': '14.3750918'}], 
#      ['Dinner-in-a-Private-Palazzino', '14 mins', 'Gozo', '1:30', {'lat': 35.8824172, 'lng': '14.5224558'}, {'lat': '35.8927509', 'lng': '14.4442306'}]],
#     [['Sne-Sculpting-Workshop', '25 mins', 'Siggiewi', '0:45', {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8628212, 'lng': 14.569725}],
#      ["Ta'-Betta-Winery-Private-Tasting", '25 mins', 'Rabat', '0:30', {'lat': 35.8542712, 'lng': 14.4212237}, {'lat': 35.8992325, 'lng': 14.5141017}],
#      ['Markus-Divinus-Private-Wine-Tasting', '10 mins', 'Valletta', '00:30', {'lat': 35.8656269, 'lng': 14.3740761}, {'lat': 35.8542712, 'lng': 14.4212237}], 
#      ['The-Xwejni-Salt-Pans', '1 hour 33 mins', None, '00:30', {'lat': 36.0803123, 'lng': 14.241213}, {'lat': 35.8656269, 'lng': 14.3740761}]]
# ]

# distance = [
#     [{'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8974216, 'lng': 14.5122298}, 
#      {'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 35.8976456, 'lng': 14.5156008}, 
#      {'lat': 35.8992325, 'lng': 14.5141017}],
    
    
#     [{'lat': 35.8997507, 'lng': 14.5123704}, {'lat': 35.9148366, 'lng': 14.444285}, 
#      {'lat': 35.8997507, 'lng': 14.5123704}],
    
#     [{'lat': 35.8537838, 'lng': 14.4385521}, {'lat': 35.8824172, 'lng': 14.5224558}],
    
#     [{'lat': 35.8992325, 'lng': 14.5141017}, {'lat': 36.0803123, 'lng': 14.241213}, 
#      {'lat': 35.8542712, 'lng': 14.4212237}, {'lat': 35.8656269, 'lng': 14.3740761}]
# ]

# def getist(self, data , val):
#     unique_data={}
#     arr = [(index, arr, idx, item) for index, arr in enumerate(data) for idx, item in enumerate(val) if item in arr[:-1]]
#     getarr = [(getdata[2], getdata[1]) for getdata in arr]
#     sorted_getarr = sorted(getarr, key=lambda x: x[0])
#     if sorted_getarr:
#         for index, values in sorted_getarr:
#             if index not in unique_data:
#                 if values not in unique_data.values():
#                     unique_data[index] = values
#     result_list = list(unique_data.values())
#     return result_list

# for dayslist in data:
#     for val in distance:
#         getist(data , val)
        
        



