import flask
import cv2
from image_to_text import get_paragraphs, image_to_text 


app = flask.Flask(__name__)

app.config["DEBUG"] = True

@app.route('/',methods=['GET'])
def home():
    image = flask.request.args.get('image')
    data = image_to_text(cv2.imread(image))
    par = get_paragraphs(data)
    return {"number_of_paragraphs":par[1]
    ,"paragraphs":par[0],"data":data}

app.run()