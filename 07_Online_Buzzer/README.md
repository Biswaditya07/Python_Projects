# Online Buzzer System Using Python, Flask, Socket.IO and Ngrok
This folder contains a real-time Python web application project focused on creating an online buzzer system for quizzes, competitions, classrooms, and live events. The application uses Flask, Flask-SocketIO, and Ngrok to allow participants to join through a public link and buzz live from any device.
The system records the first responders with accurate response time and displays the live buzz order instantly.

## Folder Name
07-Online-Buzzer

### 1. Scripts
Contains Python application files:
- online_buzzer.py

### 2. Deployment
Contains Ngrok setup details, public URL access workflow, and hosting instructions.

### 3. Reports
Contains screenshots, test results, usage notes, and event execution summaries.

## Files Included

### 1. online_buzzer.py
Python script covering:
- Flask web server creation
- Real-time communication using Flask-SocketIO
- Public deployment using Ngrok tunnel
- Host authentication system
- Start / Stop / Reset buzzer controls
- Team name registration
- Real-time buzz order tracking
- Response time calculation
- Live participant synchronization
- Browser-based buzzer interface

## Technologies Used
- Python
- Flask
- Flask-SocketIO
- Eventlet
- Pyngrok
- HTML
- JavaScript
- Socket.IO CDN

## Core Functionalities

### Participant Features
- Join from mobile or laptop browser
- Enter team name
- Press buzzer button live
- View real-time ranking
- See response times instantly

### Host Features
- Secure host key access
- Start buzzer round
- Stop buzzer round
- Reset round
- Control live event flow

### Smart Logic Included
- Only active rounds accept buzzes
- Duplicate team entries blocked
- Top first 3 responders captured
- Automatic live updates to all users
- Response timing in seconds

## Real-Time Workflow
1. Host runs Python script
2. Ngrok generates public URL
3. Participants open URL on devices
4. Host starts round
5. Teams press BUZZ
6. System records fastest responses
7. Live leaderboard updates instantly

## Skills Demonstrated
- Python Web Development
- Real-Time Systems
- Event Automation
- Flask Backend Development
- Socket Programming
- Deployment with Ngrok
- Frontend + Backend Integration
- Live Competition Tech Solutions

## Business / Event Use Case
Ideal for:
- Quiz Competitions
- College Fests
- Corporate Training Games
- Classroom Activities
- Talent Shows
- Live Audience Engagement
- Team Building Events

## Key Advantages
- No expensive hardware buzzer required
- Works online from anywhere
- Multi-device support
- Fast setup in minutes
- Accurate response tracking
- Easy host control

## Security Note
- Replace default host key before live events
- Use personal Ngrok auth token
- Rotate credentials for public competitions

## Future Improvements
- Admin login dashboard
- Team score tracking
- Unlimited participant support
- Timer countdown
- Sound effects
- Export results to Excel
- Cloud deployment on Render / Railway / AWS
- QR code join link

## Author
Biswaditya07 (Biswaditya Saha) 
MBA in Business Analytics
