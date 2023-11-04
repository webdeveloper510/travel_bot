import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz

from myapp.models import *


def read_csv(request, data,topic_id,vendor_select):
    count_10 = 0
    count_30 = 0
    dict2w="place" 
    matching_row=""
    dict_10={}
    dict_30={}
    row_has_match = False
    matching_rows=[]
    ramen = pd.read_csv('./static/media/user_csv/AI_Database_Spreadsheet_-_Sample_CSV_List_3wTNma3.csv')
  
    
    search_str = data

    # for index, row in ramen.iterrows():
   
    #     row_has_match = False
    
    #     for column_name, cell_value in row.items():
        
    #         cell_value_str = str(cell_value)
            
    #         matches = fuzz.ratio(search_str, cell_value_str)
           
            
    #         if matches >= 70:     
    #             row_has_match = True
    #             break  
    
    #     if row_has_match:
     
    #         vendor_name = row["Vendor"]
    #         matching_rows.append(vendor_name)
    # print("sssss",matching_rows)


   
    
    # for value in  matching_rows:
    #     matchesvendor1 = process.extract(value, ramen["Vendor"], limit=None)  

      
    #     for best_match1, score1,score2 in matchesvendor1:
        
       
    #         if score1 >= 90:
            
            
    #             vendor_name=ramen["Vendor"]

    #             matching_row = ramen[ramen["Vendor"] == best_match1]
               
       

    matchesvendor = process.extract(search_str, ramen["Vendor"], limit=None)  
    
    for best_match1, score1,score2 in matchesvendor:
        
       
        if score1 >= 80:
           
           
            vendor_name=ramen["Vendor"]

            matching_row = ramen[ramen["Vendor"] == best_match1]
            
           
            dict_10={
                dict2w: best_match1
            }

            dict_30={
            dict2w: best_match1
        }

    
   
    vendor_select="Falconry Experience"
   
    if dict_10 and dict_30:
        if dict_10[dict2w]!=vendor_select and dict_30[dict2w]!=vendor_select:

            vendor_select =dict_10[dict2w]
            vendor_select =dict_30[dict2w]
   
    if vendor_select:

        dict_10={
           dict2w:vendor_select
        }

        dict_30={
           dict2w:vendor_select
        }
 

    if dict_10 and dict_30:
       
        for column in list(ramen.columns):

            if type(matching_row) == str:
               
                matching_row = ramen[ramen["Vendor"] == vendor_select]

               
                
                matches = fuzz.ratio(search_str,column)  

                
                
           
                if matches >=10:
                    if str(matching_row[column].values[0]) != "nan":
                        dict[column]=matching_row[column].values[0]

                
               
                
            else:   
             
                  if not matching_row.empty:

                    matches = fuzz.ratio(search_str, column)

                    if matches >= 30:
                        
                       

                        if str(matching_row[column].values[0]) != "nan":
                            count_30 += 1
                            dict_30[column] = matching_row[column].values[0]

                    if matches >= 20:
                     

                        if str(matching_row[column].values[0]) != "nan":
                            count_10 += 1
       
    
                            dict_10[column] = matching_row[column].values[0]
    
    if "Vendor" in dict_10.keys() :
     
        del dict_10["Vendor"]
    
    if "Vendor" in dict_30.keys():

        del dict_30["Vendor"]



    if count_30 > 3:
        print("i am here")
        return dict_30
    else:
        print("i am there")
        
        return dict_10