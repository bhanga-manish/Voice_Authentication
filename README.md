# Voice_Authentication
Voice authentication prototype for AXIS bank hackathon.

The purpose of this project is to authenticate user based on user's voice.

To run this project you must install required libraries and place the saved model weights in respective folders.
And you should have two folders named "Users" and "Test"

after meeting these requirement you run command
"Python Voice_Authentication_API.py"

The service will be running on your local IP

There are two steps in process Registering and testing.
The routes are IP:5000/register for registering and IP:5000/test for testing.

The input to system is single channel wave file.

While registering the name of file is considered as the name of the person. so while testing it returns the name of registered user.

