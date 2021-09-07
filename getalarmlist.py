'''
made by ASG
'''
import requests
import json
import pandas as pd


for n in range(10):
    url = "https://www.celebalarm.com/api/getCommunityList"

    formdata ={
        'headerType': 0
        ,'languageType': 0
        ,'index': 20*n
        ,'limit': 20
        ,'query': ''
        }

    response = requests.post(url=url,data=formdata)

    data = json.loads(response.text)  

    list_head=[]
    list_sub=[]
    list_date=[]

    for i in range(20):
        header = data['contentInfos'][i]['dataInfo']['header']
        subject = data['contentInfos'][i]['dataInfo']['subject']
        date = data['contentInfos'][i]['dataInfo']['created']
        # print(header)
        # print(subject)
        # print(date)
    
        list_head.append(header)
        list_sub.append(subject)
        list_date.append(date)
        
    
    list_df = pd.DataFrame({"header": list_head, "subject" : list_sub, "date" : list_date})

list_df.append(list_df)
print(list_df)    

print("\n")

print("저장중")
list_df.to_csv('Celeb_Alarm')
print("done")
