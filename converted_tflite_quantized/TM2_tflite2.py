from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import smtplib
import argparse
import io
import time
import numpy as np
import cv2

from email.mime.image import MIMEImage
from PIL import Image
from tflite_runtime.interpreter import Interpreter

gmail_addr = "t110ab0028@ntut.org.tw" # t110ab0028@ntut.org.tw、vike11107@gmail.com
gmail_pwd = "Vike91*1202" # Vike91*1202、vike911202

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def draw_circle(image, label):
    if label == "0 mask":
        color = (255, 0, 0) # blue
    elif label == "1 no mask":
        color = (0, 255, 0) # green
    elif label == "2 thumb":
        color = (255, 0, 255) # purple
    else:
        color = (255, 255, 0) # 3 nobody yellow
#     print(image.shape)
    cv2.circle(image, (600, 50), 30, color, 5)
    
def send_gmail(gmail_addr, gmail_pwd, msg, label):
    smtp_gmail = smtplib.SMTP("smtp.gmail.com", 587)
    smtp_gmail.ehlo()
    smtp_gmail.starttls()
    smtp_gmail.login(gmail_addr, gmail_pwd)
    if label == "0 mask":
        to_addrs = "t110ab0007@ntut.org.tw"
        status = smtp_gmail.sendmail(from_addr = gmail_addr, to_addrs = to_addrs, msg = msg)
    elif label == "1 no mask":
        to_addrs = "vike11107@gmail.com"
        status = smtp_gmail.sendmail(from_addr = gmail_addr, to_addrs = to_addrs, msg = msg)
    elif label == "2 thumb":
        to_addrs = "t110ab0019@ntut.org.tw"
        status = smtp_gmail.sendmail(from_addr = gmail_addr, to_addrs = to_addrs, msg = msg)
    else:
        status = "沒人"
    if not status:
        print("寄信成功")
    else:
        print("寄信失敗", status)
    smtp_gmail.quit()
    
def get_mime_img(subject, fr, to, img):
    img_encode = cv2.imencode(".jpg", img)[1]
    img_bytes = img_encode.tobytes()
    mime_img = MIMEImage(img_bytes)
    mime_img["Content-type"] = "application/octet-stream"
    mime_img["Content-Disposition"] = 'attachment; filename = "pic.jpg"'
    mime_img["Subject"] = subject
    mime_img["From"] = fr
    mime_img["To"] = to
    return mime_img.as_string()
        
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels)

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    key_detect = 0
    times=1
    img_pre = None
    while (key_detect==0):
        ret,image_src =cap.read()
        gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        img_now = cv2.GaussianBlur(gray, (13, 13), 5)
        

        frame_width=image_src.shape[1]
        frame_height=image_src.shape[0]

        cut_d=int((frame_width-frame_height)/2)
        crop_img=image_src[0:frame_height,cut_d:(cut_d+frame_height)]

        image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

        start_time = time.time()
        if (times==1):
            results = classify_image(interpreter, image)
            elapsed_ms = (time.time() - start_time) * 1000
            label_id, prob = results[0]
            label = labels[label_id]
            draw_circle(image_src, label)
            if img_pre is not None:

                diff = cv2.absdiff(img_now, img_pre)
                ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 print(cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
                
#                 print(contours, _)
                
                if contours:
                    cv2.drawContours(image_src, contours, -1, (255, 255, 255), 2)
                    print("偵測到移動")
                else:
                    print("靜止畫面")
                    pass
                # print(label, prob)
            

        cv2.putText(image_src, label + " " + str(round(prob,3)), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

        times=times+1
        if (times>1):
            times=1
            
        cv2.imshow('Detecting....', image_src)
        img_pre = img_now.copy()
        
        k = cv2.waitKey(1)
        if k == ord("q") or k == ord("Q"):
            key_detect = 1
        elif k == ord("s") or k == ord("S"):
            print(label)
            msg = get_mime_img("拍照並傳到本人郵件", "筆電攝影機", "電子郵件", crop_img)
            send_gmail(gmail_addr, gmail_pwd, msg, label)
                                     

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#           key_detect = 1
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
  main()