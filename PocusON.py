#A반 1조
#PocusON: 온라인 수업 관리 프로그램

############PocusON############

#### FOR GRAPH ####
import pandas as pd
from matplotlib import pyplot
import datetime

#### FOR FACE IDENTIFY ####
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
from sklearn.svm import SVC
from faceidentify.SVMclassifier import model as svm
from faceidentify.SVMclassifier import out_encoder
import faceidentify.SVMclassifier

#### FOR GAZE AND MOTION ####
import argparse
import cv2
import os.path as osp
from detectheadposition import headpose
from gaze_tracking import GazeTracking

import time # For sleep
import threading #For multi thread
from datetime import date
import openpyxl

# get the face embedding for one face
def get_embedding(model, face_pixels):

   # scale pixel values
   face_pixels = face_pixels.astype('float32')
   mean, std = face_pixels.mean(), face_pixels.std()
   face_pixels = (face_pixels - mean) / std
   samples = np.expand_dims(face_pixels, axis=0)
   yhat = model.predict(samples)
   return yhat[0]

# main function
def main(args):
    filename = args["input_file"]
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    model = load_model('models/facenet_keras.h5')

    if filename is None:
        isVideo = False
        webcam = cv2.VideoCapture(0)
        webcam.set(3, args['wh'][0])
        webcam.set(4, args['wh'][1])
    else:
        isVideo = True
        webcam = cv2.VideoCapture(filename)
        fps = webcam.get(cv2.webcam_PROP_FPS)
        width = int(webcam.get(cv2.webcam_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.webcam_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Variable Setting
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])
    gaze = GazeTracking() # import gazetracking
    std_name = input("아이디를 입력하세요: ")

    prev_location = [0, 0]
    curr_location = [0, 0]
    distance = []
    photo_block = []
    arr = []
    prev_second = -1
    prev_second2 = -1
    prev_min = -1
    prev_min2 = -1
    prev_min3 = -1
    time_stamp = []
    prev_point = 0

    count = 0
    flag = True
    LookDown_time = False
    distraction_point = 0
    current_time = time.time()
    attendance_score = 100

    # Infinity Loop for Detect Cheating for Online test
    while(webcam.isOpened()):

        ret, frame = webcam.read() # Read wabcam
        gaze.refresh(frame)
        frame = gaze.annotated_frame() # Mark pupil for frame
        tm = time.localtime()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE) # face structure

        #Detect head position
        if isVideo:
            frame, angles = hpd.process_image(frame)
            if frame is None:
                break
            else:
                out.write(frame)
        else:
            frame, angles = hpd.process_image(frame)

            if angles is None :
                pass
            else :
                if angles[0] < -10 and LookDown_time == False:
                    LookDown_time = True
                    prev_min3 = tm.tm_min
                elif angles[0] > 0:
                    LookDown_time = False

                else:
                    pass
                if LookDown_time : ### 시간수정
                    curr_min = tm.tm_min
                    if curr_min - prev_min3 == 5:
                        print("고개를 들어 화면을 확인해주세요")
                        distraction_point += 1
                        LookDown_time = False
        try:
            check_empty = faces[0]
        except:
            if prev_min == -1:
                prev_min = tm.tm_min
            else:
                curr_min = tm.tm_min

                if curr_min - prev_min == 10:
                    attendance_score = attendance_score - 10
                    print("자리비움")
                    prev_min = -1

       # Draw a rectangle around the faces and predict the face name
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # take the face pixels from the frame
            crop_frame = frame[y:y+h, x:x+w] # turn the face pixels back into an image
            new_crop = Image.fromarray(crop_frame) # resize the image to meet the size requirment of facenet
            new_crop = new_crop.resize((160, 160)) # turn the image back into a tensor
            crop_frame = np.asarray(new_crop) # get the face embedding using the face net model
            face_embed = get_embedding(model, crop_frame) # it is a 1d array need to reshape it as a 2d tensor for svm
            face_embed = face_embed.reshape(-1, face_embed.shape[0]) # predict using our SVM model
            pred = svm.predict(face_embed) # get the prediction probabiltiy
            pred_prob = svm.predict_proba(face_embed) # pred_prob has probabilities of each class

            # get name
            class_index = pred[0]
            class_probability = pred_prob[0,class_index] * 100
            predict_names = out_encoder.inverse_transform(pred)
            text = '%s (%.3f%%)' % (predict_names[0], class_probability)

            #add the name to frame but only if the pred is above a certain threshold
            if (class_probability > 70):
                cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Display the resulting frame
        cv2.imshow('PocusON', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        hori_ratio = gaze.horizontal_ratio()
        verti_ratio = gaze.vertical_ratio()

        try:
            if curr_location == [0,0]:
                curr_location = [hori_ratio, verti_ratio]
            else:
                prev_location = curr_location
                curr_location = [hori_ratio, verti_ratio]
                hori_diff = curr_location[0] - prev_location[0]
                verti_diff = curr_location[1] - prev_location[1]

                if prev_second2 == -1:
                    prev_second2 = tm.tm_sec
                else:
                    curr_second2 = tm.tm_sec
                    if curr_second2 - prev_second2 == 1 or curr_second2 - prev_second2 < 0:
                        distance.append((hori_diff ** 2) + (verti_diff ** 2))
                        prev_second2 = curr_second2

                        if len(photo_block) < 3:
                            photo_block.append((hori_diff ** 2))

                # len(distance), sum(distance)임의 값 설정
                if len(distance) > 59:
                    if sum(distance) > 1: # 수정
                        print('주의 산만')
                        distraction_point += 1
                        distance = distance[1:]

        except:
            curr_location = [0.5, 0.5]

        if flag:
            if count == 10:
                flag = False
                if sum(photo_block) >= 0.0001:
                    if arr.count(std_name) > 6:
                        print(std_name, "님 출석이 완료되었습니다.")
                    else:
                        print("출석이 정상적으로 처리되지 않았습니다.")
                else:
                    print("사진으로 인식되었습니다.")

            if prev_second == -1:
                prev_second = tm.tm_sec
            else:
                curr_second = tm.tm_sec
                if curr_second - prev_second == 1 or curr_second - prev_second < 0:
                    arr.append(predict_names[0])
                    count += 1
                    prev_second = curr_second

        curr_min2 = tm.tm_min

        if prev_min2 == -1:
            prev_min2 = tm.tm_min
        else:
            curr_min2 = tm.tm_min
            if curr_min2 - prev_min2 == 10:
                date = str(time.localtime()[1]) + '/' + str(time.localtime()[2]) + " " + str(
                    time.localtime()[3]) + ":" + str(time.localtime()[4])
                if not time_stamp:
                    time_stamp.append([date, 0])
                    prev_point = distraction_point
                    prev_min2 = curr_min2
                else:
                    time_stamp.append([date, distraction_point - prev_point])
                    prev_point = distraction_point
                    prev_min2 = curr_min2

    ending_time = time.time()
    class_time = ending_time - current_time
    distract_ratio = (distraction_point / class_time) * 100
    print("귀하의 집중력은 %f입니다." % (100 -distract_ratio))
    if arr.count(std_name) > 6:
        today = datetime.date.today()
        try:
            wb = openpyxl.load_workbook('check_attendance_' + predict_names[0] + '.xlsx')

        except:
            wb = openpyxl.Workbook()

            sheet = wb.active
            sheet.append(["NAME", "DATE", "ATTEND_SCORE", "CONCENT_SCORE"])

        sheet = wb.active
        sheet.append([predict_names[0], today, attendance_score, 100 - distract_ratio])

        wb.save("check_attendance_" + predict_names[0] + ".xlsx")
        print("엑셀로 저장 되었습니다.")

        time_stamp = pd.DataFrame(time_stamp)
        pyplot.figure(figsize = (10, 5))
        pyplot.plot(time_stamp[0], time_stamp[1])
        pyplot.savefig('distract_score_' + predict_names[0] + '.png')

    # When everything done, release the webcam
    webcam.release()
    if isVideo:
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', default='gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    main(args)
