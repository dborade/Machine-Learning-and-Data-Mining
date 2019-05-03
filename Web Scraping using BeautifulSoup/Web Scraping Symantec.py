#!/usr/bin/env python
# coding: utf-8

# # Web scraping using BeautifulSoup
# 

# In[39]:


#import beautifulsoup
from bs4 import BeautifulSoup

#import requests
import requests

#Define URL
url = "https://www.symantec.com/security_response/landing/vulnerabilities.jsp"


# In[40]:


# Request content from web page
response = requests.get(url)
content = response.content


# In[41]:


soup = BeautifulSoup(content, 'lxml')
table = soup.find("table",{'id':'listings'})
tbody = table.find('tbody')
#retrieve all the rows from the table 
rows = tbody.findAll('tr')


# In[42]:


from datetime import datetime

#create lists to store retrieved data

#vulnerablility names
vul_names =[]

#discovered dates
date_values=[]

#the urls of the vulnerbility description pages
url_values = []

#in each row
for row in rows: 
    link_to_vul = row.find('a')
    vul_text = link_to_vul(text = True)[0]
    vul_names.append(vul_text)
    
    #Construct the URLs
    url_values.append("https://www.symantec.com/"+link_to_vul['href'])
        
    #date is the text of the last <td> tag
    td = row.findAll('td')
    #remove the <td></td> tags from the string
    date = str(td[-1]).replace("<td>",'').replace("</td>","")
    date = datetime.strptime(date, "%m/%d/%Y")
    #add date to the date_values list
    date_values.append(date)


# In[43]:


#Requirement 1

#use input() to get user input
from_date_input = input("Please write from date in format mm/dd/yyyy: ")
to_date_input = input("Please write to date in format mm/dd/yyyy: ")
user_term = input("Please write a term you want to find: ")

from_date = datetime.strptime(from_date_input,"%m/%d/%Y")
to_date = datetime.strptime(to_date_input,"%m/%d/%Y")

for index in range(len(vul_names)):
    if(to_date > date_values[index] > from_date):
        vul_text = vul_names[index].lower()
        if(vul_text.count(user_term) > 0):
            print(vul_names[index])    


# In[44]:


#Requirement 2
count = 0 

for index in range(100):
    response = requests.get(url_values[index])
    content =response.content
    soup = BeautifulSoup(content,'lxml')
    section = soup.find("section",{'class':'content-band'}) 
    description = section.find(text = 'Description').findNext('p')
    desc = str(description).replace("<p>",'').replace("</p>","")
    descr_to_search = desc.lower()
       
    if descr_to_search.find(user_term) != -1 :
        count += 1
        discovery_date = section.find(text ="Date Discovered").findNext('p')
        disc_date = str(discovery_date).replace("<p>",'').replace("</p>","")
        print("Found Vulnerable Number ",count," in row ",index+1, "< ",disc_date," >","\n", vul_names[index], "\n",url_values[index],"\n") 
                


# In[ ]:



    
 


# In[ ]:





# In[ ]:




