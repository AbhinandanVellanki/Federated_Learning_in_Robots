import requests
import json

url = 'http://0.0.0.0:80/'
# url = "http://127.0.0.1:5000/"
data = {'input' : 'no'}
#data = {'input' : 'How are you'}
#data = {'input' : 'What is Federated Learning'}
#data = {'input' : 'How are you feeling about the things going on'}
#data = {'input' : 'Are you smart'}
#data = {'input': 'I am very happy'}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, data=json.dumps(data), headers=headers)

print(r.text)
# payload = {'number': 2, 'value': 1}
# r = requests.post("http://127.0.0.1:5000/", data={'input'='test'})