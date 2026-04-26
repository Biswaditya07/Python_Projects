!pip install flask flask-socketio pyngrok eventlet

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from pyngrok import ngrok
import time

# 🔐 CHANGE THIS BEFORE EVENT
HOST_KEY = "12345"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

buzz_data = []          # [(team, response_time)]
start_time = None
buzzer_active = False

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Buzzer</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>

<body style="text-align:center;font-family:Arial">

<h1>🏁 Time Controlled Buzzer</h1>

<input id="team" placeholder="Enter Team Name"/>
<br><br>

<button onclick="buzz()" style="font-size:20px;padding:10px 20px;">
BUZZ 🔔
</button>

<h2>Buzz Order (with Response Time)</h2>
<ol id="results"></ol>

<hr>

<div id="hostControls" hidden>
    <button onclick="startBuzzer()" style="background:green;color:white;padding:8px 15px;">
    Start Buzzer
    </button>

    <button onclick="stopBuzzer()" style="background:orange;color:white;padding:8px 15px;">
    Stop Buzzer
    </button>

    <button onclick="reset()" style="background:red;color:white;padding:8px 15px;">
    Reset
    </button>
</div>

<script>
var socket = io();

// Identify host
let isHost = prompt("Are you the host? (yes/no)");
if(isHost && isHost.toLowerCase() === "yes"){
    document.getElementById("hostControls").hidden = false;
}

function buzz(){
    let team = document.getElementById("team").value;
    if(team === ""){
        alert("Enter Team Name");
        return;
    }
    socket.emit("buzz", team);
}

function startBuzzer(){
    let key = prompt("Enter Host Key:");
    socket.emit("start", {key:key});
}

function stopBuzzer(){
    let key = prompt("Enter Host Key:");
    socket.emit("stop", {key:key});
}

function reset(){
    let key = prompt("Enter Host Key:");
    socket.emit("reset", {key:key});
}

socket.on("update", function(data){
    let list = document.getElementById("results");
    list.innerHTML = "";
    data.forEach(function(item){
        let li = document.createElement("li");
        li.innerText = item.team + " — " + item.time + " sec";
        list.appendChild(li);
    });
});

socket.on("status", function(msg){
    alert(msg);
});
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@socketio.on("start")
def start_buzzer(data):
    global start_time, buzzer_active, buzz_data
    if data.get("key") == HOST_KEY:
        buzz_data = []
        start_time = time.time()
        buzzer_active = True
        emit("update", [], broadcast=True)
        emit("status", "Buzzer Started!", broadcast=True)

@socketio.on("stop")
def stop_buzzer(data):
    global buzzer_active
    if data.get("key") == HOST_KEY:
        buzzer_active = False
        emit("status", "Buzzer Stopped!", broadcast=True)

@socketio.on("buzz")
def handle_buzz(team):
    if buzzer_active and len(buzz_data) < 3:
        teams = [t[0] for t in buzz_data]
        if team not in teams:
            response_time = round(time.time() - start_time, 2)
            buzz_data.append((team, response_time))
            formatted = [{"team": t, "time": rt} for t, rt in buzz_data]
            emit("update", formatted, broadcast=True)

@socketio.on("reset")
def reset_buzzer(data):
    global buzz_data, buzzer_active
    if data.get("key") == HOST_KEY:
        buzz_data = []
        buzzer_active = False
        emit("update", [], broadcast=True)
        emit("status", "Buzzer Reset", broadcast=True)

# 🔑 Provide your ngrok authtoken here. Get it from https://dashboard.ngrok.com/get-started/your-authtoken
# You need to uncomment the line below and replace 'YOUR_AUTHTOKEN' with your actual ngrok token.
# For example: ngrok.set_auth_token("2f5344d5c...")
ngrok.set_auth_token("37w0YKv1mswXDMD66kxCVakxRQc_3m3PFdGb97ni6rLyXQYkt") # <-- Replace "YOUR_AUTHTOKEN" with your actual token

# 🔗 Start ngrok
public_url = ngrok.connect(5000)
print("🔗 Public URL:", public_url)

socketio.run(app, port=5000)
