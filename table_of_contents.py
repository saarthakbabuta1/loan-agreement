import cv2
import numpy as np
import pytesseract
import re
import pdf2image
import docx


def get_contours(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return cnts

def pdf_to_text(file):
    par = []
    body = ""
    try:
        pages = pdf2image.convert_from_path(pdf_path=file,dpi=200, size=(1654,2340))
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        for i in range(len(pages)):
            data = []
            filename = file+str(i) + '.png'
            pages[i].save(filename)
            image = cv2.imread(filename)
            cnts = get_contours(img=image)
                
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                cropped = image[y:y + h, x:x + w]
                

                text = pytesseract.image_to_string(cropped)

                data = [text] + data
                body = text + " " + body 
                cv2.imshow("image",cropped)
                cv2.waitKey(0)
            par.append(data)
    except Exception as e:
        print(e)
        return({"Error:",e})

    return {"body":body,"paragraph":par}

par = pdf_to_text("table_of_contents.pdf")["paragraph"]

for j in par:
    for i in j:
        i = i.strip()
        i = re.sub("\s\s+", "", i)
        i = re.sub("\n","",i)

        print(i)