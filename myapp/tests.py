from django.test import TestCase

# # Create your tests here.


import requests
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def generate_itinerary(api_key, locations, num_days):
    def get_distance_matrix(api_key, origins, destinations):
        base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            'origins': '|'.join(origins),
            'destinations': '|'.join(destinations),
            'key': api_key,
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if data['status'] == 'OK':
            rows = data['rows']
            distance_matrix = [[element['distance']['value'] for element in row['elements']] for row in rows]
            return distance_matrix
        else:
            raise Exception(f"Error: {data['status']} - {data.get('error_message', 'Unknown error')}")

    def create_data_model(locations, num_days):
        # Calculate distance matrix for all locations
        distance_matrix = get_distance_matrix(api_key, locations, locations)

        data = {}
        data['locations'] = locations
        data['num_locations'] = len(locations)
        data['num_vehicles'] = 1
        data['depot'] = 0
        data['distance_matrix'] = distance_matrix

        return data

    def find_max_distance_index(visited, distance_matrix, current_location):
        valid_indices = [i for i in range(len(distance_matrix[current_location])) if i not in visited]
        if not valid_indices:
            return None
        max_distance_index = max(valid_indices, key=lambda i: distance_matrix[current_location][i])
        return max_distance_index

    data = create_data_model(locations, num_days)
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    routing.AddDimension(transit_callback_index, 0, 1, True, 'Visits')

    visited_locations = {0}  
    current_location = 0  

    PerDayLocationDict={}
    for day in range(1, num_days + 1):
        PerDayLocationDict[f"Location {day}"]=data['locations'][current_location]
        next_location_index = find_max_distance_index(visited_locations, data['distance_matrix'], current_location)
        if next_location_index is not None:
            visited_locations.add(next_location_index)
            current_location = next_location_index

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return (manager, routing, solution)
    return PerDayLocationDict

# Example usage with latitude and longitude coordinates:
api_key = ''  # Replace with your actual API key
locations = [
    '35.9223341,14.4863325',  # hotel
    '35.82969606539233,14.441795580880497',  # haqar kim
    '35.87100627260113,14.507424579408363',  # hyo=pogeum
    '35.84172667416176,14.54411242364746',  # Marsaxlokk
    '35.83656231679167,14.524218184171236'  # Ghar Dalam
]

# Set the number of days (you may adjust as needed)
num_days = len(locations)
ans=generate_itinerary(api_key, locations, num_days)

