import face_recognition
import cv2
import pickle
import os
import imutils

cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)

data = pickle.loads(open('face_enc',"rb").read())

print("Streaming started")

videoCapture = cv2.VideoCapture(0)

while True:
    ret,frame = videoCapture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face= faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(60, 60)
    )
    rgb= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    print(encodings)
    names =  []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
    
        
        if True in matches:
            matchedIdx = [i for (i, b) in enumerate(matches) if b]
            
            counts = {}
            for i in matchedIdx:
                name = data['Name'][i]
                
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                names = name
    for ((x, y, w, h), name) in zip(face, names):
        # rescale the face coordinates
        # draw the predicted face name on the image
        print(names)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, names, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("img",frame)
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break
