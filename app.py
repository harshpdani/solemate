from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from retrieval_generation import retrievalgeneration
from ingest import ingestdata

app = Flask(__name__)

load_dotenv()

vstore=ingestdata("done")
chain=retrievalgeneration(vstore)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result=chain.invoke(input)
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run(debug= True)