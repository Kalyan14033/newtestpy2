from gevent import monkey

monkey.patch_all()

from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import time
import datetime
from werkzeug.security import check_password_hash, generate_password_hash
import jwt
from datetime import  timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event,desc
from sqlalchemy.orm import scoped_session, sessionmaker
from threading import Thread, Event
import logging
from logging.handlers import TimedRotatingFileHandler
import traceback
from flask_mail import Mail, Message


# Function to load configuration from config.json
def load_config(config_path='app_config.json'):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print("Configuration file not found.")
        raise
    except json.JSONDecodeError:
        print("Error decoding JSON in the configuration file.")
        raise




app = Flask(__name__)
config = load_config()
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = config['database']['database_uri']  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)


threads = []
running_threads = {}
executor = ThreadPoolExecutor(max_workers=8)  # Use a thread pool with max 8 workers
CORS(app, resources={r"/*": {"origins":"http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="*",async_mode='gevent')

#SMTP----email caonfiguration
app.config['MAIL_SERVER'] = config['email']['smtp_server']
app.config['MAIL_PORT'] = int(config['email']['smtp_port'])
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = config['email']['sender_email']
app.config['MAIL_PASSWORD'] = config['email']['sender_password']
app.config['MAIL_DEFAULT_SENDER'] = config['email']['sender_email']
mail = Mail(app)



#logging setup
log_handler = TimedRotatingFileHandler(
    'logs/app.log',
    when='midnight',
    interval=1,
    backupCount=7
)

log_handler.setLevel(logging.INFO)  # Ensure it captures INFO logs and above
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

#Global error handler for catching all exceptions
@app.errorhandler(Exception)
def handle_exception(error):
    # Capture the error message and traceback
    error_message = f"Error occurred: {str(error)}\nTraceback: {traceback.format_exc()}"
    
    # Trigger email automatically
    send_error_email(error_message)
    
    # Return a generic error response
    return jsonify({'message': 'An unexpected error occurred. Please try again later.'}), 500


# Send error details via email
def send_error_email(error_message):
    try:
        msg = Message("Exception Occurred in Flask App", recipients=[config['email']['recipient_email']])
        msg.body = error_message
        mail.send(msg)
    except Exception as e:
        app.logger.error(f"Error sending email: {str(e)}", exc_info=True)



#app.logger.info("This is an info message")
#app.logger.error("An error occurred", exc_info=True)

# Path to the JSON file
DATA_FILE = 'C:\\Users\\Vignesh R\\SafetyProjectMaster\\clientapp\\src\\data\\info.json'


# Initialize multiple YOLO models based on camera requirements
CAMERA_MODELS = {
    0: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//fire_smoke.pt", verbose=False),
    1: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//HD.pt", verbose=False),
    2: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//pirate_ship.pt", verbose=False),
    3: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//oilLeakage.pt", verbose=False), 
    4: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//water_leakage.pt", verbose=False),
    5: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//jumpsuit.pt", verbose=False),
    6: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//HD.pt", verbose=False),
    7: YOLO("C://Users//Vignesh R//SafetyProjectMaster//Models//shipClass.pt", verbose=False),
}

# Path to videos for each camera
CAMERA_VIDEOS = {
    0: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\FireDetection.mp4", 
    1: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\HD.mp4",
    2: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\PirateShip.mp4",
    3: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\oilTest.mp4",
    4: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\WaterLeakage.mp4",
    5: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\Jumpsuit.mp4",
    6: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\Jumpsuit.mp4",
    7: "C:\\Users\\Vignesh R\\SafetyProjectMaster\\Videos\\ShipCLass.mp4"
}

# JWT secret key
JWT_SECRET = config['credential']['jwt_secret']
# In-memory user store (hashed passwords)
users = {
    config['credential']['username']: generate_password_hash(config['credential']['password'])
}


# Define the TicketInfo model (table)
class TicketInfo(db.Model):
    __tablename__ = 'ticket_info'
    id = db.Column(db.Integer, primary_key=True)
    object_id = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(50), nullable=False)
    alert_message = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    confidence =  db.Column(db.String(100), nullable=False)
    acknowledge = db.Column(db.Integer,default = 0)

# Define the ObjectDetected model (table)
class ObjectDetected(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('ticket_info.id'))
    camera_id = db.Column(db.String(50), nullable=False)
    frame_base64 = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    ticket_info = db.relationship('TicketInfo', backref='objects', lazy=True)
    

try:
    # Create the database tables inside the app context
    with app.app_context():
        db.create_all()  # This will create the tables in your SQLite database
        session_factory = scoped_session(sessionmaker(bind=db.engine))
        Session = scoped_session(session_factory)
        print("Database tables created successfully.")

except Exception as e:
    # Handle any unexpected errors during table creation or session setup
    print(f"Error occurred while creating the database tables: {str(e)}")
    
    
    app.logger.error("An error occurred", exc_info=True)




def add_detection(ticket_data,frame_base64):
    try:
        session = Session()
        timestamp = ticket_data.get('timestamp')
        if isinstance(timestamp, (float, int)):
            timestamp = datetime.datetime.fromtimestamp(timestamp)

        ticket_info = TicketInfo(
            object_id=ticket_data['objectID'],
            class_name=ticket_data['className'],
            severity=ticket_data['severity'],
            alert_message=ticket_data.get('alertMessage'),
            confidence=ticket_data['confidence'],
            timestamp=timestamp
        )
        print(ticket_info.object_id)
        session.add(ticket_info)
        session.commit()
        print('TicketInfo record added successfully with ID:', ticket_info.id)

        object_detected = ObjectDetected(
            ticket_id=ticket_info.id,
            camera_id=ticket_data['camera_id'],
            frame_base64=frame_base64,
            timestamp=timestamp
        )
        session.add(object_detected)
        session.commit()
        print('ObjectDetected record added successfully with ID:', object_detected.id)
    except Exception as e:
        dsession.rollback()
        app.logger.error("An error occurred", exc_info=True)
        print(f"Error inserting data: {e}")
    finally:
        session.close()


def find_camera_name(id):
    try:
        cameras = load_data()
        camera_names = list(map(lambda x: x["cameraName"] if x["id"] == id else None, cameras))
        camera_names = [name for name in camera_names if name is not None]
        return camera_names[0] if camera_names else "Camera not found"
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while finding camera name: {str(e)}")
        return "Error occurred while finding camera name"


def generate_jwt_token(user):
    """Generate a JWT token for the given user."""
    try:
        token = jwt.encode({    
            'user': user,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, JWT_SECRET, algorithm="HS256")
        return token
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any errors that occur during token generation
        print(f"Error occurred while generating JWT token: {str(e)}")
        return None


def verify_jwt_token(token):
    """Verify the JWT token and return the payload if valid."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# Routes for user authentication
@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if username in users and check_password_hash(users[username], password):
            token = generate_jwt_token(username)

            return jsonify({"message": "Login successful", "token": token}), 200
        
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        return jsonify({"message": "Invalid credentials"}), 401




@app.route('/api/protected', methods=['GET'])
def protected_route():
    try:
        token = request.headers.get('Authorization')    
        if not token:
            return jsonify({"message": "Token missing"}), 403

        user_data = verify_jwt_token(token)
        if not user_data:
            return jsonify({"message": "Invalid or expired token"}), 403
    
        return jsonify({"message": f"Welcome {user_data['user']}! You have access to this protected route."})
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500




@app.route('/api/ticket_info', methods=['GET'])
def get_ticket_info():
    try:
        tickets = TicketInfo.query.all()
        ticket_data = [{
            "id": ticket.id,
            "object_id": ticket.object_id,
            "class_name": ticket.class_name,
            "severity": ticket.severity,
            "alert_message": ticket.alert_message,
            "timestamp": ticket.timestamp
        } for ticket in tickets]
        return jsonify(ticket_data)
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while retrieving ticket info: {str(e)}")
        return jsonify({"message": "An error occurred while retrieving ticket information."}), 500



@app.route('/api/acknowledge_entry/<int:ticket_id>', methods=['POST'])
def acknowledge_entry(ticket_id):
    try:
        # Fetch ticket by ticket_id using session.get()
        ticket = db.session.get(TicketInfo, ticket_id)
 
        if not ticket:
            return jsonify({"message": "Ticket not found"}), 404
       
        # Update the acknowledge field
        ticket.acknowledge = 1
 
        # Commit changes to the database
        db.session.commit()
 
        # Verify changes by logging
        print(f"Ticket {ticket_id} acknowledged. Acknowledge flag: {ticket.acknowledge}")
       
        # Return a success message
        return jsonify({"message": "Ticket acknowledged successfully!"}), 200
 
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Rollback in case of any errors and log the exception
        db.session.rollback()
        print(f"Error occurred: {str(e)}")
        return jsonify({"message": f"Error occurred: {str(e)}"}), 500



@app.route('/api/camera_based_tickets/<int:cam_id>', methods=['GET'])
def camera_basedTickets(cam_id):
    try:
        if cam_id != None:
            results = (
                db.session.query(TicketInfo, ObjectDetected)
                .join(ObjectDetected, TicketInfo.id == ObjectDetected.ticket_id )
                .filter(ObjectDetected.camera_id == cam_id)
                .filter(TicketInfo.acknowledge == 0)
                .order_by(desc(TicketInfo.timestamp))
                .all()  
            )
    

    
        data = [{
            "ticket_id": ticket.id,
            "object_id": ticket.object_id,
            "class_name": ticket.class_name,
            "severity": ticket.severity,
            "alert_message": ticket.alert_message,
            "ticket_timestamp": ticket.timestamp,
            "camera_id": obj.camera_id,
            "object_timestamp": obj.timestamp,

        } for ticket, obj in results]
    
        return jsonify(data)

    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        print(f"Error occurred while taking camera based tickets: {str(e)}")
        return jsonify({"message": "An error occurred while retrieving camera based tickets."}), 500


    


@app.route('/api/ticket_page_filter/<int:filter>', methods=['GET'])
def ticket_page_filter(filter):
    try:
        if filter == 0:
            results = (
                db.session.query(TicketInfo, ObjectDetected)
                .join(ObjectDetected, TicketInfo.id == ObjectDetected.ticket_id)
                .filter(TicketInfo.severity == 'normal')
                .filter(TicketInfo.acknowledge == 0)
                .order_by(desc(TicketInfo.timestamp))
                .all()  
            )
        
        elif filter == 1:
            results = (
                db.session.query(TicketInfo, ObjectDetected)
                .join(ObjectDetected, TicketInfo.id == ObjectDetected.ticket_id)
                .filter(TicketInfo.severity == 'danger')
                .filter(TicketInfo.acknowledge == 0)
                .order_by(desc(TicketInfo.timestamp))
                .all()  
            )
        
        elif filter == 2:
            results = (
                db.session.query(TicketInfo, ObjectDetected)
                .join(ObjectDetected, TicketInfo.id == ObjectDetected.ticket_id)
                .filter(TicketInfo.severity == 'high alert')
                .filter(TicketInfo.acknowledge == 0)
                .order_by(desc(TicketInfo.timestamp))
                .all()  
            )
        else:
            return jsonify({"message": "Invalid filter value"}), 400
        
        data = [{
            "ticket_id": ticket.id,
            "object_id": ticket.object_id,
            "class_name": ticket.class_name,
            "severity": ticket.severity,
            "alert_message": ticket.alert_message,
            "ticket_timestamp": ticket.timestamp,
            "camera_id": obj.camera_id,
            "object_timestamp": obj.timestamp,
        } for ticket, obj in results]
    
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while filtering tickets: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


    

@app.route('/api/object_detected', methods=['GET'])
def get_object_detected():
    try:
        objects = ObjectDetected.query.all()
        
        object_data = [{
            "id": obj.id,
            "ticket_id": obj.ticket_id,
            "camera_id": obj.camera_id,
            "timestamp": obj.timestamp,
            "frame_base64": obj.frame_base64
        } for obj in objects]
        
        return jsonify(object_data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while retrieving object detections: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500



@app.route('/api/ticket_with_object', methods=['GET'])
def get_ticket_with_object():
    try:
        # Perform a join query where ticket.id == object.ticket_id
        results = (
            db.session.query(TicketInfo, ObjectDetected)
            .join(ObjectDetected, TicketInfo.id == ObjectDetected.ticket_id)
            .all()
        )
    
        # Prepare the data in the required JSON format
        data = [{
            "ticket_id": ticket.id,
            "object_id": ticket.object_id,
            "class_name": ticket.class_name,
            "severity": ticket.severity,
            "alert_message": ticket.alert_message,
            "ticket_timestamp": ticket.timestamp,
            "camera_id": obj.camera_id,
            "object_timestamp": obj.timestamp,
            "frame_base64": obj.frame_base64
        } for ticket, obj in results]
    
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while retrieving tickets with objects: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500

 

@app.route('/api/detection_table_data', methods=['GET'])
def detection_table_data():
    try:
        # Perform a join query where ticket.id == object.ticket_id
        results = (
            db.session.query(TicketInfo, ObjectDetected)
            .join(ObjectDetected, TicketInfo.id == ObjectDetected.ticket_id)
            .order_by(desc(TicketInfo.timestamp))
            .all()
        )
    
        # Prepare the data in the required JSON format
        data = [{
            "ticket_id": ticket.id,
            "object_id": ticket.object_id,
            "class_name": ticket.class_name,
            "severity": ticket.severity,
            "alert_message": ticket.alert_message,
            "ticket_timestamp": ticket.timestamp,
            "camera_id": obj.camera_id,
            "object_timestamp": obj.timestamp
        } for ticket, obj in results]
    
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while retrieving detection table data: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


@app.route('/api/count_by_severity', methods=['GET'])
def count_by_severity():
    try:
        severity_counts = db.session.query(
            TicketInfo.severity,
            db.func.count(TicketInfo.id).label('count')
        ).group_by(TicketInfo.severity).all()

        result = [{
            "severity": data.severity,
            "count": data.count
        } for data in severity_counts]

        return jsonify(result)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while counting by severity: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500





@app.route('/api/ticket_object_id/<int:ticket_id>', methods=['GET'])
def ticket_object_id(ticket_id):
    try:
        # Query the TicketInfo table for the specific ticket_id
        ticket = TicketInfo.query.filter_by(id=ticket_id).first()
        
        # Check if the ticket exists
        if not ticket:
            return jsonify({"error": "No ticket found for the specified ticket_id"}), 404
        
        # Query the ObjectDetected table for objects related to this ticket_id
        obj = ObjectDetected.query.filter_by(ticket_id=ticket_id).first()

        # Prepare the data for response
        data = {
            "ticket_id": ticket.id,
            "object_id": ticket.object_id,
            "class_name": ticket.class_name,
            "severity": ticket.severity,
            "alert_message": ticket.alert_message,
            "ticket_timestamp": ticket.timestamp,
            "frame_base64": obj.frame_base64,
            "camera_id": obj.camera_id,
        }
        
        return jsonify(data)
    
    except Exception as e:
        # Handle any unexpected errors
        print(f"Error occurred while retrieving ticket by object ID: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500



@app.route('/api/report', methods=['GET'])
def get_report():
    try:
        # Aggregate count of detections by class name and severity
        report_data = db.session.query(
            TicketInfo.class_name,
            TicketInfo.severity,
            db.func.count(TicketInfo.id).label('count')
        ).group_by(TicketInfo.class_name, TicketInfo.severity).all()

        report = [{
            "class_name": data.class_name,
            "severity": data.severity,
            "count": data.count
        } for data in report_data]

        return jsonify(report)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while generating report: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500



@app.route('/api/graph_data', methods=['GET'])
def get_graph_data():
    try:
        # Aggregate count of detections per class
        graph_data = db.session.query(
            TicketInfo.class_name,
            db.func.count(TicketInfo.id).label('count')
        ).group_by(TicketInfo.class_name).all()

        graph = [{
            "class_name": data.class_name,
            "count": data.count
        } for data in graph_data]

        return jsonify(graph)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while generating graph data: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


# Load data from JSON file
def load_data():
    with open(DATA_FILE, 'r') as file:
        return json.load(file)

# Save data to JSON file
def save_data(data):
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Camera management routes
@app.route('/cameras', methods=['GET'])
def get_cameras():
    try:
        data = load_data()
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while retrieving cameras: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


@app.route('/cameras', methods=['POST'])
def add_camera():
    try:
        data = load_data()
        new_camera = request.json
        data.append(new_camera)
        save_data(data)
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while adding camera: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


@app.route('/cameras/<int:index>', methods=['PUT'])
def update_camera(index):
    try:
        data = load_data()
        updated_camera = request.json
        data[index] = updated_camera
        save_data(data)
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while updating camera: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


@app.route('/cameras/<int:index>', methods=['DELETE'])
def delete_camera(index):
    try:
        data = load_data()
        data.pop(index)
        save_data(data)
        return jsonify(data)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while deleting camera: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


@app.route('/start-all', methods=['POST'])
def start_all_cameras():
    try:
        
        global threads, running_threads
        for source, model in CAMERA_MODELS.items():
            if source in CAMERA_VIDEOS and source not in running_threads:
                input_path = CAMERA_VIDEOS[source]
                thread = CameraThread(cam_id=input_path, source=source, model=model)
                #executor.submit(thread.run)  # Submit to thread pool
                thread.start()
                running_threads[source] = thread
        return jsonify({"status": "started all cameras"})
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while starting cameras: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500



@app.route('/stop-all', methods=['POST'])
def stop_all_cameras():
    global running_threads
    try:
        for cam_id, thread in list(running_threads.items()):
            print(f"Thread {cam_id} - State: {'alive' if thread.is_alive() else 'initial or stopped'}")
            if thread.is_alive():
                print(f"Stopping camera thread: {cam_id}")
                thread.stop()  # Custom stop method
                thread.join(timeout=5)  # Wait for the thread to terminate
                print(f"Camera thread {cam_id} stopped successfully.")
            elif thread.ident is None:  # Thread in 'initial' state
                print(f"Camera thread {cam_id} was never started. Skipping.")
            else:
                print(f"Camera thread {cam_id} already stopped.")
            
            del running_threads[cam_id]  # Clean up regardless of state
        
        return jsonify({"status": "stopped all cameras"}), 200

    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        print(f"Error stopping cameras: {e}")
        return jsonify({"error": "Failed to stop all cameras", "details": str(e)}), 500


@app.route('/start-camera/<int:camera_id>', methods=['POST'])
def start_camera(camera_id):
    try:
        global CAMERA_MODELS, CAMERA_VIDEOS, threads
        if camera_id in running_threads:
            return jsonify({"status": "Camera already running"})

        model = CAMERA_MODELS.get(camera_id)
        input_path = CAMERA_VIDEOS.get(camera_id)
        
        if model and input_path:
            thread = CameraThread(cam_id=input_path, source=camera_id, model=model)
            executor.submit(thread.run)  # Use thread pool to manage camera threads
            running_threads[camera_id] = thread
            return jsonify({"status": f"Started camera {camera_id}"})
        
        return jsonify({"status": "Camera or model not found"})
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while starting camera {camera_id}: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500


@app.route('/ships', methods=['GET'])
def get_ships():
    try:
        ships = [
            {"id": 1, "name": "Ship 1", "latitude": 37.7749, "longitude": -122.4194, "location": "San Francisco", "hasDetection": False},
            {"id": 2, "name": "Ship 2", "latitude": 34.0522, "longitude": -118.2437, "location": "Los Angeles", "hasDetection": True},
            {"id": 3, "name": "Ship 3", "latitude": 40.7128, "longitude": -74.0060, "location": "New York", "hasDetection": False},
            {"id": 4, "name": "Ship 4", "latitude": 51.5074, "longitude": -0.1278, "location": "London", "hasDetection": True},
            {"id": 5, "name": "Ship 5", "latitude": 35.6895, "longitude": 139.6917, "location": "Tokyo", "hasDetection": True},
        ]
        
        return jsonify(ships)
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        # Handle any unexpected errors
        print(f"Error occurred while retrieving ships: {str(e)}")
        return jsonify({"message": "An error occurred while processing your request."}), 500

# Helper class for camera thread management

class CameraThread(threading.Thread):
    def __init__(self, cam_id, source, model):
        threading.Thread.__init__(self)
        self.source = source
        self.cam_id = cam_id
        self.model = model
        self.running = True
        self.lock = threading.Lock()  # Use a lock for thread-safe flag update
        self.tickets = []
        self._stop_event = Event()

    def run(self):
        try:
            cap = cv2.VideoCapture(self.cam_id)
            if not cap.isOpened():
                print(f"Failed to open camera {self.cam_id}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = int(fps)  # One frame per second
            frame_count = 0
            last_detection_time = {}

            while self.is_running() and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every N frames
                if frame_count % frame_interval == 0:
                    try:
                        results = self.model.track(frame)
                        annotated_frame = results[0].plot()
                        
                        
                        for result in results:
                            for box in result.boxes:
                                obj_id = int(box.id[0]) if box.id else "unknown"
                                class_id = int(box.cls[0]) if box.cls[0] is not None else -1
                                class_name = result.names[class_id] if class_id in result.names else "unknown"
                                confidence = float(box.conf[0]) * 100 if box.conf else 0
                                camera_name = find_camera_name(self.source)
                                
                                if class_name in ["NO-Hardhat", "NO-Safety Vest"]:
                                    severity = "high alert"
                                elif class_name in ["fire", "smoke", "pirate ship"]:
                                    severity = "danger"
                                else:
                                    severity = "normal"
                                
                                alert_message = f"Alert Message - {class_name} (ID: {obj_id})"
                                
                                current_time = time.time()

                                if obj_id not in last_detection_time or current_time - last_detection_time[obj_id] > 5:
                                    last_detection_time[obj_id] = current_time
                                    ticket_info = {
                                        "camera_id": self.source,
                                        "objectID": obj_id,
                                        "className": class_name,
                                        'alertMessage': alert_message,
                                        "confidence": f"{class_name} has detected in {camera_name} with confidence of : {confidence:.2f}%",  
                                        'severity': severity,
                                        'source': self.source,
                                        "timestamp": current_time
                                    }

                                    self.tickets.append(ticket_info)
                                    if class_name != "unknown":
                                        socketio.emit('object_detected', ticket_info, namespace='/')
                                        # socketio.emit('ticketData', self.tickets, namespace='/')
                                        # add_detection(ticket_info, frame_data)
                                        if ticket_info['severity'] == 'high alert':
                                            
                                            _, buffer = cv2.imencode('.jpg', annotated_frame)
                                            frame_data = base64.b64encode(buffer).decode('utf-8')
                    
                                    # Send frame via WebSocket
                                            socketio.emit(f'frame{self.source}', frame_data, namespace='/')
                                        else:
                                            
                                            frame_data = None
                                        add_detection(ticket_info,frame_data)

                    except Exception as e:
                        app.logger.error("An error occurred", exc_info=True)
                        print(f"Error processing frame for camera {self.cam_id}: {e}")

                frame_count += 1
                time.sleep(1 / fps)

        except Exception as e:
            app.logger.error("An error occurred", exc_info=True)
            print(f"Error in camera thread {self.cam_id}: {e}")
        
        finally:
            cap.release()  # Release the camera when done

    def is_running(self):
        """Check if the thread should be running in a thread-safe way."""
        with self.lock:
            return self.running

    def stop(self):
        """Stop the thread in a thread-safe way."""
        with self.lock:
            self.running = False
        self._stop_event.set()
        print(f"Camera thread {self.source} stopping...")

# Start the Flask app with WebSocket support using gevent
if __name__ == '__main__':
    try:
        from gevent.pywsgi import WSGIServer
        from geventwebsocket.handler import WebSocketHandler
        
        # Use gevent's WSGIServer to serve the app
        http_server = WSGIServer(('0.0.0.0', 7890), app, handler_class=WebSocketHandler)
        http_server.serve_forever()
    
    except Exception as e:
        app.logger.error("An error occurred", exc_info=True)
        print(f"Error starting the Flask app: {e}")
