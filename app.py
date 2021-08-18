from flask import Flask, render_template, request
import pickle


# Use pickle to load in the pre-trained model.
with open(f'model.pkl', 'rb') as f:
    model = pickle.load(f)
app=Flask(__name__,template_folder='templates')

@app.route('/', methods=["GET","POST"])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == "POST":
        Pregnancies= request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        
       

        input_variables = pd.DataFrame([[Pregnancies,Glucose,BloodPressure,SkinThickness
        ,Insulin,BMI,DiabetesPedigreeFunction,Age]],columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'
       ])

        prediction = model.predict(input_variables)
       
        if prediction == 0:
            result = 'NO'
        else:
            result = 'YES'

        return render_template('main.html',result=prediction)


if __name__ == '__main__':
    app.run()