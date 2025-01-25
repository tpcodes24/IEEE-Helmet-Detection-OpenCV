from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandastable import Table, TableModel
from PIL import Image, ImageTk
import tkinter.messagebox as tm
import cv2
import time
from sklearn.externals import joblib #library for load model
from skimage.feature import hog
from scipy import ndimage
import warnings
import smtplib
import random
import string
import mysql.connector
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")




vehicle_classifier = joblib.load('vehicle.pkl') #Path for Vehicle Classifier
helmet_classifier = joblib.load('helmet.pkl')   #Path for Helmet classifier
LARGE_FONT=("Veradna",12)
count=0
fgbg=cv2.createBackgroundSubtractorMOG2()


def createHogDescriptor(image):
    H = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(2, 2),transform_sqrt=True,visualize=False,multichannel=True)
    return H

def hide_area(mask_frame):
        contours,hierarchy = cv2.findContours(mask_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(original_image,contours,-1,(0,255,0),5)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt)>50000 or h<w or cv2.contourArea(cnt)<3000:
                cv2.fillPoly(mask_frame,[cnt],(0,0,0))
        return mask_frame

def extract_images(mask,real,totalCount,totalBikeCount,totalHelmetCount,prevTotalCount,prevBikeCount,prevHelmetCount):
    currTotalCount=0
    currBikeCount=0
    currHelmetCount=0
    
    height,width,_=real.shape
    
    add=30
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
       # m=cv2.moments(cnt)
       # cx=int(m['m10']/m['m00'])
       # cy=int(m['m01']/m['m00'])
        if h>w and y>=(height-height//3) and y<=(height-height//3+add) and cv2.contourArea(cnt)>500:
            roi = real[y:y+h,x:x+w+1]
            temp = roi.copy()
            roi = cv2.resize(roi,(80,160))
            roi = createHogDescriptor(roi)
            ans = vehicle_classifier.predict(roi[np.newaxis,:])
            ans = int(ans)    
            if ans==2:
                continue
            wearHelmet=False
            if ans==1:  #If object is bike
                currBikeCount+=1
                temp =  real[y-3:y+h+3,x:x+w+1].copy()
                temp = temp[:temp.shape[0]//3,:]
                H = createHogDescriptor(cv2.resize(temp,(80,160)))
                ans = int(helmet_classifier.predict(H[np.newaxis,:]))
        
                if ans==0:  #iIf object has helmet
                    cv2.rectangle(real,(x,y),(x+w,y+h),(0,255,0),2) 
                    wearHelmet=True
                    currHelmetCount+=1
            currTotalCount+=1
            if wearHelmet==False:
                cv2.rectangle(real,(x,y),(x+w,y+h),(0,0,255),2)
    
    if currTotalCount>prevTotalCount:
        totalCount+=currTotalCount-prevTotalCount
    prevTotalCount=currTotalCount
    
    if currBikeCount>prevBikeCount:
        totalBikeCount+=currBikeCount-prevBikeCount
    prevBikeCount=currBikeCount
    
    if currHelmetCount>prevHelmetCount:
        totalHelmetCount+=currHelmetCount-prevHelmetCount
    prevHelmetCount=currHelmetCount
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(real,'Total Count:'+str(totalCount),(10,40), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(real,'Bike Count:'+str(totalBikeCount),(10,80), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(real,'Helmet Count:'+str(totalHelmetCount),(10,120), font, 0.5,(255,255,255),2,cv2.LINE_AA)
                
    cv2.line(real,(0,height-height//3),(width-1,height-height//3),(255,255,255),2)
    cv2.line(real,(0,height-height//3+add),(width-1,height-height//3+add),(255,255,255),2)
    
    return real,totalCount,totalBikeCount,totalHelmetCount,prevTotalCount,prevBikeCount,prevHelmetCount 
    
"""Traffic class"""
class Traffic(Tk):
    def __init__(self,*args,**kwargs):
        Tk.__init__(self,*args,**kwargs)
        Tk.wm_title(self,"Traffic_Management") #set the titile name
        container=Frame(self)
        container.pack(side="top",fill="both",expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        #menubar
        menubar=Menu(self)
        homemenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Home",menu=homemenu)
        homemenu.add_command(label="Home",command=lambda:self.show_frame(HomePage))
        homemenu.add_command(label="Logout",command=self._login_btn_clicked)

        viewmenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=viewmenu)
        viewmenu.add_command(label="Data View",command=lambda:self.show_frame(DataViewPage))
        viewmenu.add_command(label="Graphical View",command=lambda:self.show_frame(GraphPage))

        aboutmenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="About", menu=aboutmenu)
        aboutmenu.add_command(label="Help",command=lambda:self.show_frame(HelpPage))
        aboutmenu.add_command(label="About",command=lambda:self.show_frame(AboutPage))
        Tk.config(self,menu=menubar)
        self.bind('<Escape>', lambda e: self.destroy())
        
        #store frame in Frames
        
        self.frames={}
        
        for f in (HomePage,DataViewPage,GraphPage,AboutPage,HelpPage):
            frame=f(container,self)
            self.frames[f]=frame
            frame.grid(row=0,column=0,sticky="nsew")
        self.show_frame(HomePage) #show the homepage
        
    #change the frames
    def show_frame(self,cont):
        
        frame=self.frames[cont]
        frame.tkraise()
    #call the login class    
    def _login_btn_clicked(self):
        self.destroy()
        app=Login()
        app.state("zoomed")
        #app.geometry("1350x750+0+0")

       
"""Homepage class this class show the livestreaming"""        
class HomePage(Frame):
    
    def update_data(self,date,totalvehicle,with_helmet,without_helmet):
        mydb = mysql.connector.connect(
        host="localhost",
        user="tejasree",
        passwd="parasa",
        database="vehicle")
        mycursor = mydb.cursor()
        sql="INSERT INTO surviellience_record_tbl(date,bike_with_helmet,bike_without_helmet,total_vehicle) VALUES(%s,%s,%s,%s)"
        val=(date,with_helmet,without_helmet,totalvehicle)
        mycursor.execute(sql,val)
        mydb.commit()
    
    def get_frame(self):
        
        self.cap=cv2.VideoCapture('c2.avi')
        
        
        #self.cap=cv2.VideoCapture("Car_6.mp4")
        canvas = Canvas(self.frame_video, width =500, height =500,bg='cadet blue')#set the GUI canvas size
        canvas.grid(row=15,column=3,sticky='news')
        button1=Button(self.frame_label,text="Stop",command=self.cap.release,bd=5, height = 1, width =15)
        button1.grid(row=1,column=2)
        ret,first_frame=self.cap.read()
        ret,second_frame=self.cap.read()
        
        totalCount=0
        totalBikeCount=0
        totalHelmetCount=0
        prevTotalCount=0
        prevBikeCount=0
        prevHelmetCount=0
        endtime=datetime.now()+timedelta(seconds=30)
        
        while True:
            
            
            #ret,frame=self.cap.read() #read frame/image
            if ret:
                
                mask_frame=cv2.cvtColor(cv2.absdiff(first_frame,second_frame),cv2.COLOR_BGR2GRAY)
                mask_frame = cv2.resize(mask_frame,(500,500))
                original_frame=first_frame
                original_frame=cv2.resize(original_frame,(500,500))
               # mask_frame=fgbg.apply(first_frame)
                mask_frame= cv2.medianBlur(mask_frame,5)
                mask_frame= cv2.medianBlur(mask_frame,5)

                ret,mask_frame = cv2.threshold(mask_frame,40,255,cv2.THRESH_BINARY)
                kernel = np.ones((5,5),np.uint8)    
                mask_frame=cv2.dilate(mask_frame,kernel,iterations=3) #increase
                mask_frame=cv2.erode(mask_frame,np.ones((3,8),np.uint8),iterations=1) #decrease
                mask_frame=cv2.morphologyEx(mask_frame,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
                height,width,_=original_frame.shape

                mask_frame=hide_area(mask_frame) 
                frame,totalCount,totalBikeCount,totalHelmetCount,prevTotalCount,prevBikeCount,prevHelmetCount=extract_images(mask_frame,original_frame,totalCount,totalBikeCount,totalHelmetCount,prevTotalCount,prevBikeCount,prevHelmetCount)

                first_frame=second_frame
                ret,second_frame=self.cap.read()
                
                #database upadte#
                withoutHelmet=totalBikeCount-totalHelmetCount
                currtime=datetime.now()
                diff=endtime-currtime
                if(diff.seconds==0):
                    self.update_data(currtime,totalCount,totalHelmetCount,withoutHelmet)
                    endtime=datetime.now()+timedelta(seconds=30)
                
                
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(500,500))
                #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert the image in grayscale
                photo =ImageTk.PhotoImage(image =Image.fromarray(frame),master=self) #convert the image from array to original image
                canvas.create_image(0, 0, image=photo, anchor=NW) #display the image inside the canvas of GUI

                #update the GUI window    
                self.update_idletasks()
                self.update()
            else:
                break
        self.update_idletasks()
        self.update()
    
    
    def __init__(self,parent,controller):
        Frame.__init__(self,parent)
        self.config(bg='powder blue')
        self.container = Frame(self,relief="solid",bg='powder blue')
        self.container.pack()
        #self.cap=cv2.VideoCapture()
        self.lbl=Label(self.container,text='| LIVE TRAFFIC STREAMING|',bg='powder blue',font=('arial',20,'bold'))
        self.lbl.grid(row=0,column=0,columnspan=2,pady=40)
        self.frame_label=Frame(self.container,bd=0,relief='ridge',bg='cadet blue')
        self.frame_label.grid(row=1,column=0)
        button=Button(self.frame_label,text="Start Live Streaming",bd=5,command=self.get_frame, height = 1, width =15)
        button.grid(row=1,column=0)    
        self.frame_video=Frame(self.container,bd=10,relief='ridge',bg='cadet blue')
        self.frame_video.grid(row=2,column=0)
        

"""data view class this class shows the data in table form""" 
class DataViewPage(Frame):
    def pri(self):
        print("dataview")
    def __init__(self,parent,controller):
        Frame.__init__(self,parent)
        mydb = mysql.connector.connect(
        host="localhost",
        user="tejasree",
        passwd="parasa",
        database="vehicle")
        mycursor = mydb.cursor()
        mycursor.execute('SELECT * FROM surviellience_record_tbl')
        #df = pd.read_sql('SELECT * FROM surviellience_record_tbl')
        table_rows = mycursor.fetchall()
        df = pd.DataFrame(table_rows, columns=mycursor.column_names)
        pt = Table(self, dataframe=df,showtoolbar=True, showstatusbar=True)
        pt.show()
        self.pri()
        
"""grappage class this class represent the data on graph"""       
class GraphPage(Frame):
    def __init__(self,parent,controller):
        Frame.__init__(self,parent)
        self.config(bg='powder blue')
        self.frame=Frame(self,bg='powder blue')
        self.frame.pack()
        plt.style.use("ggplot")
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        
        mydb = mysql.connector.connect(
        host="localhost",
        user="tejasree",
        passwd="parasa",
        database="vehicle")
        mycursor = mydb.cursor()
        mycursor.execute('SELECT * FROM surviellience_record_tbl')
        table_rows = mycursor.fetchall()
        df2 = pd.DataFrame( [[ij for ij in i] for i in table_rows])
        a.clear()         
        dta=df2.iloc[-1]
        y_pos=np.arange(3)
        per=[dta[1],dta[2],int(dta[3])]
        p1=a.bar(y_pos,per,align='center',alpha=0.5,color=['red','blue','green'])
        #a.set_xticklabels(['Bike Rider with Helmet','Bike Rider without helmet','Total Vehicles'],rotation=0)
        
        a.set_ylabel('count')
        a.set_xticklabels([])
        a.legend((p1[0],p1[1],p1[2]),('Bike Rider with Helmet','Bike Rider without helmet','Total Vehicles'))
        
        #title="Graph Between X and Y"
        #a.set_title(title)
        canvas = FigureCanvasTkAgg(f, self.frame)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
        
"""Aboutpage class this class shows full information of software"""
class AboutPage(Frame):
    def __init__(self,parent,controller):
        Frame.__init__(self,parent)
        self.config(bg='powder blue')
        text="""         Welcome to About section\n\n 
        In the perception to make a traffic and accident system acute, we provide desktop application which   ''
        can reduces the effort of traffic police and provide moreover functionality to them for catch the
        bikers easily who don’t wear helmets. Desktop application will automatically extract the images of
        bikers who don’t wear helmets and it will save these images for further approach. Images will be 
        extracts from the surveillance video on public road. Desktop application will provide features like
        visualization of data related to occurrence of vehicles on different attributes.
"""
        label=Label(self,text=text,font=('arial',20,'bold'),bg='powder blue',justify=LEFT)
        label.pack(pady=10,padx=10)
        
"""HelpPage class this class shows the FAQ for the software"""
class HelpPage(Frame):
    def __init__(self,parent,controller):
        Frame.__init__(self,parent)
        self.config(bg='powder blue')
        text="""       Help section!\n\n
        1. Home page shows the livestreaming of footage.
        1.1 Click on livestreaming button.
        1.2 If you want to close the application click on stop button first.
        2.1 In View section data view and graph view.
        2.1 Click on Data view it show data in tabular form.
        2.2 Click on Graphical view it shows the data in graph form.
        3. Help page
        4. About Page this descirbe the how the system work.
        """
        label=Label(self,text=text,font=('arial',20,'bold'),bg='powder blue',justify=LEFT)
        label.pack(pady=10,padx=10)

        
def main():
    app=Traffic()
    app.state("zoomed")
    #app.geometry("1280x720")
    
    
    
"""login class"""
class Login(Tk):
    
    def table(self):
        mydb = mysql.connector.connect(
        host="localhost",
        user="tejasree",
        passwd="parasa",
        database="vehicle")
        mycursor = mydb.cursor()
        tbl='surviellience_user_tbl'
        _SQL = """SHOW TABLES"""
        mycursor.execute(_SQL)
        results=mycursor.fetchall()
        results_list=[item[0] for item in results]
        if tbl not in results_list:
            mycursor.execute("CREATE TABLE surviellience_record_tbl (date DATETIME primary key, bike_with_helmet BIGINT(8),bike_without_helmet BIGINT(8),total_vehicle VARCHAR(20))")
            mycursor.execute("CREATE TABLE surviellience_user_tbl (username VARCHAR(35) NOT NULL,passwd VARCHAR(20) NOT NULL)")
            sql="INSERT INTO surviellience_user_tbl(username,passwd) VALUES(%s,%s)"
            val=("admin","passwd")
            mycursor.execute(sql,val)
            mydb.commit()
        else:
            print("already exist")
            
    def __init__(self,*args,**kwargs):
        Tk.__init__(self,*args,**kwargs)
        Tk.wm_title(self,"Traffic_Management")
        self.config(bg='powder blue')
        self.bind('<Escape>', lambda e: self.destroy())
        self.frame=Frame(self,bg='powder blue')
        self.frame.pack()
        """label"""
        self.lbl=Label(self.frame,text='|WELCOME To Traffic And Management System|',bg='powder blue',font=('arial',20,'bold'))
        self.lbl.grid(row=0,column=0,columnspan=2,pady=40)#.pack()
        
        self.frame_label=Frame(self.frame,bd=10,relief='ridge',bg='cadet blue')
        self.frame_label.grid(row=1,column=0)
        
        """username password"""
        self.lbl_username=Label(self.frame_label,text='Username: ',bg='cadet blue',font=('arial',10,'bold'))
        self.lbl_username.grid(row=0,column=0)#.pack()
        self.entry_username=Entry(self.frame_label)
        self.entry_username.grid(row=0,column=1)
        self.lbl_password=Label(self.frame_label,text='Password: ',bg='cadet blue',font=('arial',10,'bold'))
        self.lbl_password.grid(row=1,column=0)
        self.entry_password=Entry(self.frame_label,show="*")
        self.entry_password.grid(row=1,column=1)
        
        """check button"""
        self.checkbox = Checkbutton(self.frame_label, text="Keep me logged in",bg='cadet blue')
        self.checkbox.grid(row=2,columnspan=2)
        
        """login button"""
        self.logbtn = Button(self.frame_label, text="Login", bd=3,command=self._login_btn_clicked,bg='cadet blue',font=('arial',10,'bold'))
        self.logbtn.grid(row=3,column=0)
        self.logbtn = Button(self.frame_label, text="Forget Password", bd=3,command=self.change_passwd,bg='cadet blue',font=('arial',10,'bold'))
        self.logbtn.grid(row=3,column=1)
        
    #fuction for screen detroy
    def destroy_login(self):
        self.destroy()
        
    #fuction for username and password validation
    def _login_btn_clicked(self):
        self.table()
        username = self.entry_username.get()
        password = self.entry_password.get()
        mydb = mysql.connector.connect(
        host="localhost",
        user="tejasree",
        passwd="parasa",
        database="vehicle")
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM surviellience_user_tbl")
        myresult = mycursor.fetchall()
        for x in myresult:
            if x[0]==username and x[1]==password:
                self.destroy_login()
                main()
            else:
                tm.showerror("Login error", "Invalid username/password")
      
        
       
        
    def send(self,u,p):
        sender=''
        receiver=u
        password=''
        smtpserver=smtplib.SMTP('smtp.gmail.com',587)
        smtpserver.ehlo()
        smtpserver.starttls()
        smtpserver.ehlo
        smtpserver.login(sender,password)
        msg='Subject:Username and Password of traffic mangagement system\nDear Customer,\n\n'
        msg+='Your password and username has been changed.\nThe new password and username are:\nUsername='
        msg+=u
        msg+='\nPassword='
        msg+=p
        smtpserver.sendmail(sender,receiver,msg)
        smtpserver.close()
    
    def id_generator(self,size=4, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def verify(self):
        self.user=self.entry_changed_username.get()
        #self.email=self.entry_changed_email.get()
        self.email=self.user
        self.passwd=self.id_generator()
        mydb = mysql.connector.connect(
        host="localhost",
        user="tejasree",
        passwd="parasa",
        database="vehicle")
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM surviellience_user_tbl")
        myresult = mycursor.fetchall()
        for x in myresult:
            if x[0]==self.user:
                sql = "UPDATE surviellience_user_tbl SET username= %s WHERE username= %s"
                val = (self.email, x[0])
                mycursor.execute(sql, val)
                mydb.commit()
                sql = "UPDATE surviellience_user_tbl SET passwd= %s WHERE username= %s"
                val = (self.passwd,x[0])
                mycursor.execute(sql, val)
                mydb.commit()
                Label(self.frame_label, text = "Sucessfully updated", fg = "green", font =["calibri",11]).grid(row=4,columnspan=1)
                self.send(self.email,self.passwd)
            else:
                Label(self.frame_label, text = "You are not Authorized person", fg = "green", font =["calibri",11]).grid(row=4,columnspan=3)
                
        
        
        
    def change_passwd(self):
        top=Toplevel(self)
        top.config(bg='powder blue')
        self.frame=Frame(top,bg='powder blue')
        self.frame.pack()
        """label"""
        self.lbl=Label(self.frame,text='|WELCOME To Traffic And Management System|',bg='powder blue',font=('arial',20,'bold'))
        self.lbl.grid(row=0,column=0,columnspan=2,pady=40)#.pack()
        
        self.frame_label=Frame(self.frame,bd=10,relief='ridge',bg='cadet blue')
        self.frame_label.grid(row=1,column=0)
        
        """username password"""
        self.lbl_changed_username=Label(self.frame_label,text='Old Username: ',bg='cadet blue',font=('arial',10,'bold'))
        self.lbl_changed_username.grid(row=0,column=0)#.pack()
        self.entry_changed_username=Entry(self.frame_label)
        self.entry_changed_username.grid(row=0,column=1)
        self.logbtn = Button(self.frame_label, text="submit", bd=3,command=self.verify,bg='cadet blue',font=('arial',10,'bold'))
        self.logbtn.grid(row=3,columnspan=3)
        
app=Login()
app.state("zoomed")
app.mainloop()