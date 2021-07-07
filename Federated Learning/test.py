import requests
import json

url = 'http://192.168.0.103:80/'
# url="http://127.0.0.1:5000/"
# payload = {"input": "doing something"}
# # self.log(payload)
# r=requests.post(url, data=payload)
# print(r.text)

data = {'input': 'Hello'}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, data=json.dumps(data), headers=headers)

print(r.text)
# payload = {'number': 2, 'value': 1}
# r = requests.post("http://127.0.0.1:5000/", data={'input'='test'})