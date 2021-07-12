import re

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','doc','docx'])

COVENANTS = ["CONDITIONS PRECEDENT","SECURITY", "REPESENTATIONS AND WARRANTIES", 
"COVENANTS", "TAXES", "EVENT OF DEFAULT","PREPAYMENT"]

def allowed_file(filename):
	try:
		print("Check for valid file extension")
		return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	except Exception as e:
		return(e)

def regex(text):
	text = re.sub("\s\s+", "",text)
	text = re.sub("\n","",text)
	text = re.sub("|","",text)
	text = re.sub("\u201c","",text)
	text = re.sub("\u201d","",text)
	text = re.sub("\u2014","",text)
	text = re.sub("\u2018","",text)
	text = text.strip()
	return text

def is_heading(text):
	if text in COVENANTS:
		return "covenant_headings"
