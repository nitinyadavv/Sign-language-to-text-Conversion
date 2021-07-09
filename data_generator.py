import cv2
import os
import time


# Creating the Directory Structure for storing all image classes(if not present already)
if not os.path.exists("data"):
    os.makedirs("data")
 
    for i in range(65,91):
        os.makedirs(f'data/{chr(i)}')

    for i in range(65,91):
        os.makedirs(f'data/{chr(i)}')

# Function to Extract useful Features from ROI by applying Different Filters and Techniques
def convert(frame):
    minValue = 70
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res



# MODE OF CURRENT WINDOW
_MODE=" DATA GENERATION "
directory="data/"

capture=cv2.VideoCapture(0)

while True:
    ret,frame=capture.read()
    frame=cv2.flip(frame,1)
    if ret==False:
        continue

    #Storing # of Images in Each Label    
    count = [len(os.listdir(f'{directory}/{chr(i)}')) for i in range(65,91)]    
    
    cv2.putText(frame, f'--- MODE : {_MODE.capitalize()} ---', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
    cv2.putText(frame, "--- IMAGE COUNT ---", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
    
    
    #Displaying Count of All Labels
    x_coordinate=10
    y_coordinate=80
    for i in range(65,91):
        cv2.putText(frame,f' {chr(i)} : {count[i-65]} ',(x_coordinate,y_coordinate),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        y_coordinate+=15



    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])


    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0) ,1)
 
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (128, 128)) 
 
    cv2.imshow("Frame", frame)
    
    #Extracing and Converting
    roi=convert(roi)

    cv2.imshow("ROI", roi)
    
    #Detecting Key Interrupts
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == 27: # esc key
        break
    
    # Save Image under label and Update
    else:
        for i in range(65,91):
            if (interrupt & 0xFF == i) or (interrupt & 0xFF == i+32):
                cv2.imwrite(f'{directory}/{chr(i)}/{count[i-65]}.jpg', roi)
                count[i-65]+=1
                    
capture.release()
cv2.destroyAllWindows()




