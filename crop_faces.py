import cv2
import sys
import os
import glob

class FaceCropper(object):
    CASCADE_PATH = "./haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.3, 5, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            i += 1
            cv2.imwrite("./results/"+image_path.split('/')[-1].split('.')[0]+"_%d.jpg" % i, faceimg)


if __name__ == '__main__':
    detecter = FaceCropper()
    imgList = glob.glob('./data/*')
    for img in imgList:   
        print (img)     
        detecter.generate(img, True)
