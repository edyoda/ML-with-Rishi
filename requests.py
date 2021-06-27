# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 20:11:57 2021

@author: RISHBANS
"""

import requests
import json

ip_address= "52.66.207.90"
port = "5000"
data = [[5.8, 2.8, 5.1, 2.4]]

url = 'http://{0}:{1}/predict/'.format(ip_address, port)

json_data = json.dumps(data)
header = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
response = requests.post(url, data=json_data, headers = header)
print(response, response.text)