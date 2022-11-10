import json
import sys

try:
    block = json.loads('{"name":"Leanne Graham","email":"Sincere@april.biz"}')
    if block['name']:
       output = block['name']
    else:
       output = "No Name Found."
except:
    output = "No Name Found"

