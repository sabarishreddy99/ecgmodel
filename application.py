from flask import Flask,render_template,request
import Output

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def getvalue():
    ecg=request.form['ecgvals']
    ecg=str(ecg)
    res=Output.getoutput(ecg)
    return render_template('index.html',outputstr=res)

if __name__=="__main__":
    app.run()