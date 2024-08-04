import os

data_path = ""

print(os.getcwd())

if os.path.exists(data_path):
    print("e")
else:
    print("no")