import face_recognition
from imutils import paths
import cv2
import os
import pickle

cascPathface = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

img = input("Enter the name of imagefolder: ")
knownEncodings = []
knownNames = []
imagepath = list(paths.list_images("images/"+img))
iname = input("Enter the Name of Image: ")
for(i, imagepath) in enumerate(imagepath):
    name = iname 
    image = cv2.imread(imagepath)
    image = cv2.resize(image,(620,540))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        # rescale the face coordinates
        # draw the predicted face name on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("img",image)
    cv2.waitKey(0)
    rgb  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model="hog")
    encodings = face_recognition.face_encodings(rgb,boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
data = { "Name" : knownNames, "encodings" : knownEncodings}

f = open("face_enc", "a")
f.write(pickle.dumps(data))
f.close()