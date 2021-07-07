
# client2.py
#!/usr/bin/env python

import socket
import time 
import os
import pickle 
TCP_IP = '192.168.0.103'
TCP_PORT = 9001
BUFFER_SIZE = 1024
import encrypt


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

# filename='client_model.pt'
filename = 'model.h5'
encrypt.encrypt('model.h5')
encFilename = 'model_e.h5'
fsize = str(os.path.getsize(encFilename))
print("fsize: " + fsize)

s.send(fsize.encode("ascii").strip())
time.sleep(2)
f = open(encFilename,'rb')
l = f.read(BUFFER_SIZE)
while (l):
    s.send(l)
    l = f.read(BUFFER_SIZE)
f.close()
time.sleep(1)


print('Successfully sent the file to server')
with open('received.h5', 'wb') as f:
    print ('file opened')
    while True:
        #print('receiving data...')
        data = s.recv(BUFFER_SIZE)
        # print('data=%s', (data))
        if not data:
            print ('file close()')
            f.close()
            break
        # write data to a file
        f.write(data)
encrypt.decrypt('received.h5')

print('Successfully recived the file from server')
s.close()
print('connection closed')
