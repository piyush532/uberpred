import numpy as np
from flask import Flask,request,jsonify,render_template
#request=for api generation #jsonify=api data ke lie hai
#render_template=for attaching html
import pickle#to import model
import math

#creating object
app=Flask(__name__)
model=pickle.load(open('taxi.pkl','rb'))

#now we have to have our html pages to it

@app.route('/')#this is by default used for home page
def home():
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
	int_features=[int(x)for x in request.form.values()]
	final_features=[np.array(int_features)]
	prediction=model.predict(final_features)
	output=round(prediction[0],2)
	return render_template('index.html',prediction_text="Number of weekly Rides should be {}".format(math.floor(output)))




if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8080)#in devlopment phase we always

