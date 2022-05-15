import cv2
import face_recognition

img1 = face_recognition.load_image_file('swarajAnand.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1test = face_recognition.load_image_file('swaraj.jpeg')
img1test = cv2.cvtColor(img1test,cv2.COLOR_BGR2RGB)

face = face_recognition.face_locations(img1)[0] #y1,x2,y2,x1
faceTest = face_recognition.face_locations(img1test)[0]
# print(face)
encodeFace = face_recognition.face_encodings(img1)[0]
encodeTestFace = face_recognition.face_encodings(img1test)[0]
# print(encodeFace)
cv2.rectangle(img1, (face[3],face[0]),(face[1],face[2]),(0,0,255),3)
cv2.rectangle(img1test, (faceTest[3],faceTest[0]),(faceTest[1],faceTest[2]),(0,0,255),3)

res = face_recognition.compare_faces([encodeFace],encodeTestFace)
# print(res)

face_dis = face_recognition.face_distance([encodeFace],encodeTestFace)
# print(dis)

cv2.putText(img1test, f"{res} {round(face_dis[0],2)}",(50,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,50),3)

cv2.imshow("ORIGINAL",img1)
cv2.imshow("TEST IMG",img1test)
cv2.waitKey()
cv2.destroyAllWindows()