import cv2
import numpy as np
import pytesseract
import re
import pdf2image

#file = "GHPL Loan Agreement-edited.pdf"
file = "image.png"
if file.rsplit('.', 1)[1].lower() == 'pdf':
    pages = pdf2image.convert_from_path(pdf_path=file,dpi=200, size=(1654,2340))
    data = ""
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    for i in range(len(pages)):
        filename = file+str(i) + '.png'
        pages[i].save(filename)
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        data = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            cropped = image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            data.append(text)
        cv2.imshow('thresh', thresh)
        cv2.imshow('dilate', dilate)
        cv2.imshow('image', image)
        cv2.waitKey()
else:

    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    data = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        cropped = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        data.append(text)
    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('image', image)
    cv2.waitKey()

for i in data:
    i = re.sub("\s\s+", " ", i)
    i = i.strip()
    if len(i)>3:
        print(i)
    