import pickle

data = []

a = {"Id":1,"Name":"Gaurav","Branch":"CSE"}

data.append(a)

f=open("file","wb")
f.write(pickle.loads(data))
f.close