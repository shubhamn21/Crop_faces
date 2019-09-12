import cv2
import sys
import os
import glob

class FaceCropper(object):
    CASCADE_PATH = "./haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result):
        cap = cv2.VideoCapture(image_path)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        path = "./results/"+image_path.split('/')[-1].split('.')[0]
        if not os.path.exists(path):
            os.makedirs(path)
        idx=0
        seconds = 1
        fps = cap.get(cv2.CAP_PROP_FPS) 
        multiplier = fps * seconds
        print ("Processing ...")
        while True:
          frameId = int(round(cap.get(1)))
          flag, img = cap.read()
          idx+=1
          if frameId % multiplier == 0:   
            faces = self.face_cascade.detectMultiScale(img, 1.3, 5, minSize=(100, 100))
            if (faces is None):
                print('Failed to detect face')
                return 0

            facecnt = len(faces)
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
                cv2.imwrite(path+"/"+image_path.split('/')[-1].split('.')[0]+"_original_%d_%d.jpg" % (idx/10,i), img)
                cv2.imwrite(path+"/"+image_path.split('/')[-1].split('.')[0]+"_cropped_%d_%d.jpg" % (idx/10,i), faceimg)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break


if __name__ == '__main__':
    detecter = FaceCropper()
    imgList = glob.glob('./data/*')
    for img in imgList:   
        print (img)     
        detecter.generate(img, True)
