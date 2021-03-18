import face_recognition
import os
import cv2
import imutils
import pickle

cascPathface = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

data = pickle.loads(open('face_enc',"rb").read())
imageName = input("Enter the image name you detect : ")
image = cv2.imread("images/random/"+imageName)
image = cv2.resize(image,(620,540))

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
faceloc = face_recognition.face_locations(rgb,model="hog")
encodings = face_recognition.face_encodings(rgb, faceloc)
print(encodings)
names = "unknown"
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
        
# loop over the recognized faces
for ((x, y, w, h), name) in zip(faces, names):
        # rescale the face coordinates
        # draw the predicted face name on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, names, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

while True:
    cv2.imshow("Frame", image)
    key = cv2.waitKey(0) & 0xff
    if key == 27:
        break    