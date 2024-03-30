#**************************************************************************************

#

#   Driver Monitoring Systems using AI (code sample)

#

#   File: eyes_position.m

#   Author: Jacopo Sini

#   Company: Politecnico di Torino

#   Date: 19 Mar 2024

#

#**************************************************************************************



# 1 - Import the needed libraries

import cv2

import mediapipe as mp

import numpy as np 

import time

import statistics as st

import os


def perclos(times):
    return (times[2]-times[1])/(times[3]-times[0])

def check_alarm_10s(time_state1):
    if(time_state1 >= 10 ):
        print("ALARM: EAR > 80% for more than 10s")

def eye_yaw(pupil_xy, centre_xy, eye_width):
    return (pupil_xy[0]-centre_xy[0])*90/(eye_width/2)

def eye_pitch(pupil_xy, centre_xy, eye_width):
    return (pupil_xy[1]-centre_xy[1])*90/(eye_width/2)


# 2 - Set the desired setting

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

    max_num_faces=1,

    refine_landmarks=True, # Enables  detailed eyes points

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)

mp_drawing_styles = mp.solutions.drawing_styles

mp_drawing = mp.solutions.drawing_utils



drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



# Get the list of available capture devices (comment out)

#index = 0

#arr = []

#while True:

#    dev = cv2.VideoCapture(index)

#    try:

#        arr.append(dev.getBackendName)

#    except:

#        break

#    dev.release()

#    index += 1

#print(arr)



# 3 - Open the video source

cap = cv2.VideoCapture(0) # Local webcam (index start from 0)


state = 0
perclos_times = [0, 0, 0, 0]

# 4 - Iterate (within an infinite loop)

while cap.isOpened(): 

    

    # 4.1 - Get the new frame

    success, image = cap.read()     

    start = time.time()

    # Also convert the color space from BGR to RGB

    if image is None:
        break
        #continue

    #else: #needed with some cameras/video input format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace

    image.flags.writeable = False    

    # 4.2 - Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    #The camera matrix
    focal_length=1*img_w
    cam_matrix=np.array([ [focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0,0,1]])


    point_33 = [] # Right Eye Right
    point_145 = [] # Right Eye Bottom
    point_133 = [] # Right Eye Left
    point_RET = [] # Right Eye Top

    point_362 = [] # Left Eye Right
    point_374 = [] # Left Eye Bottom
    point_263 = [] # Left Eye Left
    point_386 = [] # Left Eye Top

    point_468 = [] # Right Eye Iris Center
    point_473 = [] # Left Eye Iris Center

    face_2d=[]#aggiunte
    face_3d=[]
    left_eye_2d=[]
    left_eye_3d=[]
    right_eye_2d=[]
    right_eye_3d=[]


    # 4.3 - Get the landmark coordinates
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                # Eye Gaze (Iris Tracking)
                # Left eye indices list
                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
                # Right eye indices list
                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                #LEFT_IRIS = [473, 474, 475, 476, 477]
                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 33:
                    point_33 = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 133:
                    point_133 = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 144:
                    point_144 = (lm.x * img_w, lm.y * img_h)

                if idx == 145:
                    point_145 = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 153:
                    point_153 = (lm.x * img_w, lm.y * img_h)

                if idx == 158:
                    point_158 = (lm.x * img_w, lm.y * img_h)

                if idx == 159:
                    point_159 = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 160:
                    point_160 = (lm.x * img_w, lm.y * img_h)

                if idx == 263:
                    point_263 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 362:
                    point_362 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 373:
                    point_373 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 374:
                    point_374 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 380:
                    point_380 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 385:
                    point_385 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 386:
                    point_386 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 387:
                    point_387 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 468:
                    point_468 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 255, 0), thickness=-1)                    

                if idx == 469:
                    point_469 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 470:
                    point_470 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 471:
                    point_471 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 472:
                    point_472 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 473:
                    point_473 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 255), thickness=-1)

                if idx == 474:
                    point_474 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 475:
                    point_475 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 476:
                    point_476 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 477:
                    point_477 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x,y]) #aggiunte
                    face_3d.append([x,y,lm.z])

                #LEFT_IRIS = [473, 474, 475, 476, 477]
                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386: # iris points
                #if idx == 473 or idx == 474 or idx == 475 or idx == 476 or idx == 477: # eye border
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)                   

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    left_eye_2d.append([x,y]) #aggiunte
                    left_eye_3d.append([x,y,lm.z])

                    #EAR da rendere percentuale (massimo valore 20 da rendere a 100 per avere 80%) -> lunghezza 3 volte dell'altezza
                    #prendere indici da slide degli occhi - time con end-start e 


                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159: # iris points
                # if idx == 468 or idx == 469 or idx == 470 or idx == 471 or idx == 472: # eye border
                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)                    

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    right_eye_2d.append([x,y]) #aggiunte
                    right_eye_3d.append([x,y,lm.z])            

            # 4.4. - Draw the positions on the frame
            l_eye_width = point_263[0] - point_362[0]
            l_eye_height = point_374[1] - point_386[1]
            l_eye_center = [(point_263[0] + point_362[0])/2 ,(point_374[1] + point_386[1])/2]
            #cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=int(horizontal_threshold * l_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            cv2.circle(image, (int(point_473[0]), int(point_473[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("Left eye: x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)))
            cv2.putText(image, "Left eye:  x = " + str(np.round(point_473[0],0)) + " , y = " + str(np.round(point_473[1],0)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

            r_eye_width = point_133[0] - point_33[0]
            r_eye_height = point_145[1] - point_159[1]
            r_eye_center = [(point_133[0] + point_33[0])/2 ,(point_145[1] + point_159[1])/2]

            #cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=int(horizontal_threshold * r_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 

            cv2.circle(image, (int(point_468[0]), int(point_468[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris
            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)))

            cv2.putText(image, "Right eye: x = " + str(np.round(point_468[0],0)) + " , y = " + str(np.round(point_468[1],0)), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)              
            # speed reduction (comment out for full speed)
            #time.sleep(1/25) # [s]

        face_2d=np.array(face_2d, dtype=np.float64)
        face_3d=np.array(face_3d, dtype=np.float64)
        left_eye_2d=np.array(left_eye_2d, dtype=np.float64)
        left_eye_3d=np.array(left_eye_3d, dtype=np.float64)
        right_eye_2d=np.array(right_eye_2d, dtype=np.float64)
        right_eye_3d=np.array(right_eye_3d, dtype=np.float64)

        #The distorsion parameters
        dist_matrix=np.zeros((4,1), dtype=np.float64)

        #Solve PnP
        success_face, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        #Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        #Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # PITCH & YAW
        pitch = angles[0]*1800
        yaw = -angles[1]*1800
        roll = 180 + (np.arctan2(point_33[1] - point_263[1], point_33[0] - point_263[0])*180/np.pi)
        if roll > 180:
            roll = roll - 360
        yaw_left_eye = eye_yaw(left_pupil_2d, l_eye_center, l_eye_width)
        yaw_right_eye = eye_yaw(right_pupil_2d, r_eye_center, r_eye_width)
        pitch_left_eye = eye_pitch(left_pupil_2d, l_eye_center, l_eye_width)
        pitch_right_eye = eye_pitch(right_pupil_2d, r_eye_center, r_eye_width)

        pitch_tot = pitch + (pitch_left_eye+pitch_right_eye)/2
        yaw_tot = yaw + (yaw_left_eye+yaw_right_eye)/2

        #if (abs(pitch_tot) > 30 or abs(yaw_tot) > 30):
            #print("ALARM: driver distracted")

        #Display directions
        nose_3d_projections, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] - yaw*10), int(nose_2d[1] - pitch*10))
        cv2.line(image, p1, p2, (255,0,0),3)   

        #left_eye_3d_projections, jacobian = cv2.projectPoints(left_pupil_3d, rot_vec_left_eye, trans_vec_left_eye, cam_matrix, dist_matrix)
        p1_1 = (int(left_pupil_2d[0]), int(left_pupil_2d[1]))
        p2_1 = (int(left_pupil_2d[0] + yaw_left_eye*10), int(left_pupil_2d[1] + pitch_left_eye*10))
        cv2.line(image, p1_1, p2_1, (255,0,0),3)  
        
        #right_eye_3d_projections, jacobian = cv2.projectPoints(right_pupil_3d, rot_vec_right_eye, trans_vec_right_eye, cam_matrix, dist_matrix)
        p1_2 = (int(right_pupil_2d[0]), int(right_pupil_2d[1]))
        p2_2 = (int(right_pupil_2d[0] + yaw_right_eye*10), int(right_pupil_2d[1] + pitch_right_eye*10))
        cv2.line(image, p1_2, p2_2, (255,0,0),3) 


        #----------------------------------------------------------------------------------------------------------------------
        #EAR RIGHT EYE
        ear_right_eye = (abs(point_158[1]-point_153[1]) + abs(point_160[1]-point_144[1])+abs(point_159[1]-point_145[1]))/(3*(abs(point_33[0]-point_133[0])))

        #----------------------------------------------------------------------------------------------------------------------
        #EAR LEFT EYE
        ear_left_eye = (abs(point_385[1]-point_380[1]) + abs(point_387[1]-point_373[1])+abs(point_386[1]-point_374[1]))/(3*(abs(point_362[0]-point_263[0])))

        #MEAN EAR
        ear = (ear_left_eye+ear_right_eye)/2
        perc_open = ear*350     #300 is a tuning parameter   

        # MEASURE TIME SINCE LAST FRAME

        end = time.time()
        totalTime = end-start

        #--------------------------------------------------------------------------------------------------------------------
        # MANAGE STATE MACHINE
        if(perc_open >= 80): #state 1
            if(state == 4): #end of perclos cycle
                print("perclos: ", perclos(perclos_times)) #take all the four times and compute perclos
                state = 1
                perclos_times = [0, 0, 0, 0]

            else :
                perclos_times[0] += totalTime
                check_alarm_10s(perclos_times[0])
                state = 1

        elif(20 <= perc_open < 80):    #state 2 or 4
            if(state == 0 or state == 1): #since the previous state was 1, now we're in state 2
                state = 2
                perclos_times[1] += totalTime
            elif(state == 3): #since the previous state was 3, now we're in state 4
                state = 4
                perclos_times[3] += totalTime

        else: #perc_open < 20 => state 3
            state = 3
            perclos_times[2] += totalTime


        # ---------------------------------------------------------------------------------------------------------------------
        # istructions to be left at the end of the loop

        if totalTime>0:
            fps = 1 / totalTime

        else:
            fps=0

        
        #print("FPS:", fps)
        cv2.putText(image, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)      

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly soruce and eventual log file

cap.release()

#log_file.close()

    

# [EOF]
