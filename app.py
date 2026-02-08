import librosa
import numpy as np
import pandas as pd
import joblib
from urllib.request import urlopen
from flask import Flask, jsonify, request
from pydub import AudioSegment
import tempfile

model_url = 'https://dl.dropboxusercontent.com/scl/fi/dg18q5d6q0uef9o0mbiyf/model_gbc.joblib?rlkey=spl7kljva84sgrdjddjp1agxn&dl=0'
# model_url = 'https://dl.dropboxusercontent.com/scl/fi/onw7xu8abzmvu0i3fpmq0/model_gbc.pkl?rlkey=6dtg3a2r4880vtobvsvkai1ex&dl=0'
# model_url = 'https://dl.dropboxusercontent.com/scl/fi/e56vd17mbbcg9qa8twz5d/instrument_recognition_model.pkl?rlkey=3ctz2m8jtpdpoe7i4bvdkjp81&dl=0'
model_filename = 'model.joblib'
classes = ['piano', 'violin', 'acoustic guitar', 'cello']


def mfcc_features(signal, sample_rate):
    return np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20).T, axis=0).tolist()


def audio_preprocessing(file_path):
    df = pd.DataFrame()
    y, sr = librosa.load(file_path, sr=None)
    if y.size != 0:
        df_new = pd.DataFrame(
            data=[[str(mfcc_features(y, sr))]], columns=['MFCC_Features'])
        df = pd.concat([df, df_new], ignore_index=True)
        print(f"{file_path} features extracted")

    return df


def mfcc_df(df):
    data = pd.DataFrame()
    for index in range(df.shape[0]):
        data = pd.concat([data, pd.DataFrame([float(feature.strip().replace(
            '[', '').replace(']', '')) for feature in df.iloc[index, 0].split(',')]).transpose()])
        data.index = range(data.shape[0])
    return data


def prepare_file(audio_file):
    df = audio_preprocessing(audio_file)
    data = mfcc_df(df)
    return data


def download_model():
    response = urlopen(model_url)
    with open(model_filename, 'wb') as file:
        file.write(response.read())
    print(f'Model downloaded from {model_url} and saved as {model_filename}')


def predict_label(audio_file):
    download_model()
    final_model = joblib.load(model_filename)
    new_data = prepare_file(audio_file)
    predictions = final_model.predict_proba(new_data)
    result = ""
    for i in range(len(classes)):
        result += f"{classes[i]} {predictions[0][i] * 100:.3f}%\n"
    # result = "piano 0.124% \n violin 0.006% \n acoustic guitar 25.585% \n cello 74.285%"
    print(f'RESULT: {result}')

    return result

def convert_to_wav(mp3_file):
    wav_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    AudioSegment.from_mp3(mp3_file).export(wav_file.name, format="wav")
    return wav_file.name


app = Flask(__name__)


@app.route('/')
def index():
    return '..'


@app.route('/upload', methods=['POST'])
def upload_file():
    print(request)
    pred = ""
    response_data = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            response_data['message'] = "Error, no file"
            response_data['result'] = ""

            return jsonify(response_data), 400

        file = request.files['file']

        if file and (file.filename.endswith(('.wav')) or file.filename.endswith(('.mp3'))):
            if file.filename.endswith('.mp3'):
                converted_file_path = convert_to_wav(file)
                file.close()
                file = open(converted_file_path, 'rb')

            predicted_label = predict_label(file)
            pred += predicted_label
            response_data['result'] = pred
            response_data['message'] = "OK"

            return jsonify(response_data), 200

        response_data['message'] = "Error, bad file"
        response_data['result'] = ""

        return jsonify(response_data), 400

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
