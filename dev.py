"""Gunicorn *development* config file"""

# Flask WSGI application path in pattern MODULE_NAME:VARIABLE_NAME
swgi_app = "app:app"
# The granularity of Error log outputs
loglevel = "debug"
# The number of worker processes for handling requests
workers = 2
# The socket to bind
bind = "0.0.0.0:5000"
# Restart workers when code changes (development only!)
reload = True
# Write access and error info to /var/log
accesslog = "/home/frappe/face_recognition/face_recognition/dev.log"
# Redirect stdout/stderr to log file
capture_output = True
# PID file so you can easily fetch process ID
pidfile = "/home/frappe/face_recognition/face_recognition/dev.pid"
# Daemonize the Gunicorn process (detach & enter background)
daemon = True
