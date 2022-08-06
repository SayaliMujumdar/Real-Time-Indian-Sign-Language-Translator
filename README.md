# Real-Time-Indian-Sign-Language-Translator
In this Project I have tried creating a real time indian sign language translator.
The translator translates the gestures performed infront of the webcam.
I have used CNN for alphabet recognition and Google's mediapipe for numbers recognition.
I have tried implemting the CNN model with the help of parameters from the reference paper.

Screenshots of Result:



![Screenshot (406)](https://user-images.githubusercontent.com/61038454/183245274-1b6116e1-eb45-463b-8e5f-aa3785deb4b3.png)

![Screenshot (413)](https://user-images.githubusercontent.com/61038454/183245309-890eedd1-0fe2-43a5-8ed8-c7cc61209511.png)



Requirements:
opencv-python==4.5.5.62

tensorflow==2.8.0

mediapipe==0.8.9.1

Pillow==9.0.1

gTTS==2.2.3

playsound==1.2.2

numpy==1.22.2

matplotlib==3.5.1

imageio==2.16.1

pip==22.0.3

attrs==21.4.0

wheel==0.36.2

cryptography==36.0.2

Jinja2==3.0.3

setuptools==57.0.0


Reference Paper:
Sarkar, A., Talukdar, A.K., Sarma, K.K. (2020). CNN-Based Real-Time Indian Sign Language Recognition System. 
In: Chillarige, R., Distefano, S., Rawat, S. (eds) Advances in Computational Intelligence and Informatics. ICACII 2019. Lecture Notes in Networks and Systems, vol 119. Springer, Singapore.
https://doi.org/10.1007/978-981-15-3338-9_9

Dataset:
For alphabets:
https://sites.google.com/a/gauhati.ac.in/ece/indian-sign-language-databse
