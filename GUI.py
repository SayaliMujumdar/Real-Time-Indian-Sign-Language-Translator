from PIL import Image, ImageTk
import tkinter as tk
import cv2
from gtts import gTTS
from playsound import playsound
import  os
import tensorflow as tf
import mediapipe as mp


def get_number_of_hands(filepath):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    img = cv2.imread(filepath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image",imgRGB)
    # cv2.waitKey(200)
    n = 0
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
            n = n + 1
    else:
        print("No Hands")
    return n


def prepare(filepath):
    IMG_SIZE = 28
    img = cv2.imread(filepath)

    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


class Application:

    def __init__(self):
        self.directory = 'FinalYearProject/Code'
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.croped_image = None
        self.current_symbol ='-'
        self.flag= False

        #load model
        self.loadmodel = tf.keras.models.load_model(
                "C:/Users/91992/PycharmProjects/FinalYearProject/venv/cnnpreprocessed1.model")

        print("Loaded model from the disk .....!")
        self.root = tk.Tk()
        self.root.title("Indian Sign language to voice converter...!")

        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("700x700")
        self.root.config(bg="#f2f2f2")
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=150, width=500, height=500)
        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=650, y=150, width=350, height=350)
        self.T = tk.Label(self.root)
        self.T.place(x=250, y=5)
        self.T.config(text="Indian Sign Language Translator", font=("PT Serif",40,"bold"),fg="#336699",bg="#f2f2f2")

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=750, y=500)

        self.btnstart = tk.Button(self.root,command = self.video_loop_alpha, height=0, width=30)
        self.btnstart.config(text="Recognise Alphabets",fg ="white",bg="#336699")
        self.btnstart.place(x=50, y=100)

        self.btnstart1 = tk.Button(self.root, command=self.video_loop_digi, height=0, width=30)
        self.btnstart1.config(text="Recognise Numbers", fg="#ffffff",bg="#336699")
        self.btnstart1.place(x=250, y=100)

        self.btnhelp = tk.Button(self.root, command=self.help, height=0, width=30)
        self.btnhelp.config(text="Help", fg="white",bg="#336699")
        self.btnhelp.place(x=450, y=100)

        self.btnabout = tk.Button(self.root, command=self.about, height=0, width=30)
        self.btnabout.config(text="About", fg="white",bg="#336699")
        self.btnabout.place(x=650, y=100)

        self.btnstop = tk.Button(self.root, command=self.destructor, height=0, width=30)
        self.btnstop.config(text="Exit", fg="white",bg="#336699")
        self.btnstop.place(x=850, y=100)

        self.btnstop2 = tk.Button(self.root, command=self.make_flag_true, height=0, width=30)
        self.btnstop2.config(text="Stop predicting", fg="white", bg="#336699")
        self.btnstop2.place(x=1050, y=100)

        #self.video_loop()
    def make_flag_true(self):
        self.flag=True

    def video_loop_alpha(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame,1)
            x1 = 200
            y1 = 100
            x2 = 550
            y2 = 450
            cv2.rectangle(frame, (x1-1, y1-1), (x2, y2), (0, 255, 0), 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2image = cv2image[100:450, 100:450] #100:450, 100:450
            cv2.imwrite("test_image.jpg", cv2image)
            #cv2.imshow("test_image.jpg", cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.croped_image = cv2image

            self.current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (35, 35), 0)
            ret, res = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=str(self.current_symbol), font=("PT Serif", 100),fg="#003366")
            if self.flag:
                self.flag = False
                return
            else :
                self.root.after(1, self.video_loop_alpha)

    def predict(self, test_image):
        CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        test_image = cv2.resize(test_image, (28, 28))
        #test_image = cv2.resize(test_image,(28,28))
        cv2.imwrite("predict_image.jpg", test_image)
        if get_number_of_hands("test_image.jpg") != 0:
            result = self.loadmodel.predict(prepare("predict_image.jpg"))
            print(result)
            prediction = {}
            inde = 0
            for i in range(25):
                if result[0][inde] == 1:
                   print(CATEGORIES[inde])
                   language = 'en'
                   myobj = gTTS(text=CATEGORIES[inde], lang=language, slow=False)
                   myobj.save("welcome.mp3")
                   playsound("welcome.mp3")
                   os.remove("welcome.mp3")
                   #converttoaudio(CATEGORIES[inde])
                   break
                prediction[i] = result[0][inde]
                inde += 1
            # LAYER 1
            self.current_symbol = CATEGORIES[inde]
        else:
            self.current_symbol = "-"

    def video_loop_digi(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image  = cv2.flip(frame,1)
            x1 = 100
            y1 = 100
            x2 = 450
            y2 = 450
            cv2.rectangle(frame, (x1-1, y1-1), (x2, y2), (0, 255, 0), 1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2image = cv2image[y1:y2, x1:x2] #100:450, 100:450
            cv2.imwrite("test_image.jpg", cv2image)
            #cv2.imshow("test_image.jpg", cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.croped_image = cv2image

            self.current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            self.current_symbol = self.predict_num(cv2image)
            self.current_image2 = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (35, 35), 0)
            ret, res = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 100),fg="#003366")
            language = 'en'
            myobj = gTTS(text=str(self.current_symbol), lang=language, slow=False)
            myobj.save("welcome.mp3")
            playsound("welcome.mp3")
            os.remove("welcome.mp3")

            if self.flag:
                self.flag = False
                return
            else:
                self.root.after(1, self.video_loop_digi)


    def predict_num(self,test_image):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        mpDraw = mp.solutions.drawing_utils
        #imgRGB = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("image",imgRGB)
        # cv2.waitKey(200)
        n = 0
        bindx = 0
        bmid = 0
        bthmb = 0
        bring = 0
        bpinky = 0
        results = hands.process(test_image)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = test_image.shape
                    indxvalue = int(handLms.landmark[0].y * h) - int(handLms.landmark[8].y * h)
                    thmbvalue = int(handLms.landmark[0].x * w) - int(handLms.landmark[4].x * w)
                    midvalue = int(handLms.landmark[0].y * h) - int(handLms.landmark[12].y * h)
                    ringvalue = int(handLms.landmark[0].y * h) - int(handLms.landmark[16].y * h)
                    pinkyvalue = int(handLms.landmark[0].y * h) - int(handLms.landmark[20].y * h)
                   # print(indxvalue)
                   # print(thmbvalue)
                   # print(midvalue)
                   # print(ringvalue)
                   # print(pinkyvalue)
                    val = 0
                    if (indxvalue < 150):
                        bindx=0
                        cv2.putText(test_image, str("INDEX : CLOSED"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    else:
                        bindx=1
                        cv2.putText(test_image, str("INDEX :OPEN"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    if (midvalue < 150):
                        bmid = 0
                        cv2.putText(test_image, str("MID : CLOSED"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    else:
                        bmid = 1
                        cv2.putText(test_image, str("MID :OPEN"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    if (thmbvalue < 100):
                        bthmb = 0
                        cv2.putText(test_image, str("THUMB: CLOSED"), (10, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    else:
                        bthmb = 1
                        cv2.putText(test_image, str("THUMB :OPEN"), (10, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    if (ringvalue < 150):
                        bring = 0
                        cv2.putText(test_image, str("RING: CLOSED"), (10, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    else:
                        bring = 1
                        cv2.putText(test_image, str("RING :OPEN"), (10, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    if (pinkyvalue < 150):
                        bpinky = 0
                        cv2.putText(test_image, str("PINKY: CLOSED"), (10, 190), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    else:
                        bpinky = 1
                        #cv2.putText(img, str("PINKY :OPEN"), (10, 190), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    mpDraw.draw_landmarks(test_image, handLms, mp_hands.HAND_CONNECTIONS)
                    #print("bindx", bindx)
                    #print("bpinky", bpinky)
                    #print("bring", bring)
                    #print("bthmb", bthmb)
                    #print("bmid", bmid)
                    if bindx == 1 and bpinky == 0 and bring==0 and bthmb == 0 and bmid == 0:
                        print("1")
                        val =1

                    elif bindx ==1 and bpinky==0 and bring==0 and bthmb == 0 and bmid == 1 :
                        print(2)
                        val = 2

                    elif bindx == 1 and bpinky == 0 and bring == 0 and bthmb == 1 and bmid == 1:
                        print(3)
                        val = 3

                    elif bindx == 1 and bpinky == 1 and bring == 1 and bthmb == 0 and bmid == 1:
                        print(4)
                        val = 4

                    elif bindx == 1 and bpinky == 1 and bring == 1 and bthmb == 1 and bmid == 1:
                        print(5)
                        val = 5

                    elif bindx == 1 and bpinky == 0 and bring == 1 and bthmb == 0 and bmid == 1:
                        print(6)
                        val = 6

                    elif bindx == 1 and bpinky == 1 and bring == 0 and bthmb == 0 and bmid == 1:
                        print(7)
                        val = 7


                    elif bindx == 1 and bpinky == 1 and bring == 1 and bthmb == 0 and bmid == 0:
                        print(8)
                        val = 8

                    elif bindx == 0 and bpinky == 1 and bring == 1 and bthmb == 0 and bmid == 1:
                        print(9)
                        val = 9
                    else:
                        val = 0
                    return val
               # cv2.imshow("test",test_image)
               # cv2.waitKey(200)

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()
        cv2.destroyAllWindows()

    def destructor2(self):
        print("Closing Application...")
        self.root2.destroy()
        cv2.destroyAllWindows()


    def help(self):
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("Help")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")

        # img = cv2.imread("Pictures/sir.jpg", 1)
        # # img = cv2.resize(img, (300, 300))
        # cv2.imwrite("Pictures/sir.png", img)
        # return

        self.tx = tk.Label(self.root1)
        self.tx.place(x=330, y=20)
        self.tx.config(text="Indian Sign Language Gestures", fg="#336699", font=("Courier", 20, "bold"))

        self.photo1 = tk.PhotoImage(file='img.png')
        self.w1 = tk.Label(self.root1, image=self.photo1,bg="white")
        self.w1.place(x=20, y=105)

    def about(self):
        self.root2 = tk.Toplevel(self.root)
        self.root2.title("About")
        self.root2.protocol('WM_DELETE_WINDOW', self.destructor2)
        self.root2.geometry("700x700")

        # img = cv2.imread("Pictures/sir.jpg", 1)
        # # img = cv2.resize(img, (300, 300))
        # cv2.imwrite("Pictures/sir.png", img)
        # return

        self.tx = tk.Label(self.root2)
        self.tx.place(x=330, y=220)
        self.tx.config(text="About us : ", fg="#336699", font=("Courier", 20, "bold"))

        self.tx1 = tk.Label(self.root2)
        self.tx1.place(x=200, y=270)
        self.tx1.config(text="", fg="#336699", font=("Courier", 15, "bold"))

        self.tx2 = tk.Label(self.root2)
        self.tx2.place(x=200, y=320)
        self.tx2.config(text="", fg="#336699", font=("Courier", 15, "bold"))

        self.tx3 = tk.Label(self.root2)
        self.tx3.place(x=200, y=370)
        self.tx3.config(text="", fg="#336699", font=("Courier", 15, "bold"))

        self.tx4 = tk.Label(self.root2)
        self.tx4.place(x=200, y=420)
        self.tx4.config(text="", fg="#336699", font=("Courier", 15, "bold"))

        self.photo1 = tk.PhotoImage(file='img_1.png')
        self.w1 = tk.Label(self.root2, image=self.photo1, bg="white")
        self.w1.place(x=100, y=10)


print("Starting Application...")
pba = Application()
pba.root.mainloop()