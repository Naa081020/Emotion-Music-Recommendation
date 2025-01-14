from flask import Flask, render_template, Response, jsonify
import gunicorn
from camera import *

app = Flask(__name__)

headings = ("Name","Artist","Link")
df1 = music_rec()
df1 = df1.head(15)
@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)

def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

def music_rec():
# Assume your CSV has 'Name', 'Link', 'Artist' columns
    df = pd.read_csv(music_dist[show_text[0]])
    print(df.head())  # Debugging to see the columns
    return df[['Name', 'Link', 'Artist']]  # Changed 'Album' to 'Link'


if __name__ == '__main__':
    app.debug = True
    app.run()
