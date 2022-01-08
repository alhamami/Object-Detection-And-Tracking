import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk


path = ""
classNumber=0

#Root Config.
root =tk.Tk()
root.geometry("300x137")
root.title('Object Tracking')
root.resizable(0, 0)
root.iconphoto(False, tk.PhotoImage(file='jalal.png'))
root.eval('tk::PlaceWindow . center')
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)


#object class
objectlabel = tk.Label(root, text="Enter Class Number:", font="Raleway")
objectlabel.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
ojectEntry = Entry(root)
ojectEntry.grid(column=1 ,row=1, sticky=tk.E, padx=5, pady=5)
ojectEntry.insert(END, 'For all classes type -1')


#Video Label
videoLabel = tk.Label(root, text="Select Video:", font="Raleway")
videoLabel.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

#Browse Video Button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:open_file(), font="Raleway", bg="#20bebe", fg="white", height=1, width=10)
browse_text.set("Browse")
browse_btn.grid(column=1, row=2)



#Instructions
def instructions(event=None):
    messagebox.showinfo("INSTRUCTIONS",
                        "You must choose the video for which you want to get results and then select the type of class you want to get results for which are as follows:\n\n-1 --> For All Classes\n0 --> Person\n"
                        "1 --> Bicycle\n2 --> Car\n3 --> Motorcycle\n4 --> Airplane\n5 --> Bus\n6 --> Train\n7 --> Truck\n8 --> Boat\n9 --> Traffic Light\n10 --> Fire Hydrant\n11 --> Stop Sign\n12 --> Parking Meter\n"
                        "13 --> Bench\n14 --> Bird\n15 --> Cat\n16 --> Horse\n17 --> Sheep\n18 --> Cow\n19 --> Elephant\n20 --> Bear\n21 --> Zebra\n22 --> Giraffe\n23 --> Backpack\n24 --> Umbrella\n"
                        "25 --> Handbag\n26 --> Tie\n27 --> Suitcase\n28 --> Frisbee\n29 --> Skis\n30 --> Snowboard\n31 --> Sports Ball\n32 --> Kite\n33 --> Baseball Bat\n34 --> Baseball Glove\n35 --> Skateboard\n"
                        "36 --> Surfboard\n37 --> Tennis Racket\n38 --> Bottle\n39 --> Wine Glass\n40 --> Cup\n41 --> Fork\n42 --> Knife\n43 --> Spoon\n44 --> Bowl\n45 --> Banana\n46 --> Apple\n47 --> Sandwich\n"
                        "48 --> Orange\n49 --> Broccoli\n50 -> Carrot\n51 --> Hot dog\n52 --> Pizza\n53 --> Donut\n54 --> Cake\n55 --> Chair\n56 --> Couch\n57 --> Potted Plant\n58 --> Bed\n59 --> Dining Table\n"
                        "60 --> Toilet\n61 --> TV\n62 --> Laptop\n63 --> Mouse\n64 --> Remote\n65 --> Keyboard\n66 --> Cell Phone\n67 --> Microwave\n68 --> Oven\n69 --> Toaster\n70 --> Sink\n71 --> Refrigerator\n"
                        "72 --> Book\n73 --> Clock\n73 --> Vase\n74 --> Scissors\n75 --> Teddy Bear\n76 --> Hair Drier\n77 --> Toothbrush")
root.bind("<Button-1>", instructions())

load = Image.open("info.png")
resized_image= load.resize((25,25), Image.ANTIALIAS)
render = ImageTk.PhotoImage(resized_image)
img = Label(image=render, height=20, width=20)
img.image = render
img.place(x=2, y=110)
img.bind('<Button-1>', instructions)







# Get Result Function
def getResult():
    global classNumber
    try:
        if path == "":
            messagebox.showwarning("Select Video First",
                                 "You have to choose a video first to get the results")
        else:
            value = int(ojectEntry.get())
            if value == -1:
                value = None
            classNumber = value
            messagebox.showinfo("Video Processing",
                                "Please wait a while for the video to be processed and then the results will be displayed")
            root.destroy()
    except ValueError:
        messagebox.showerror("Enter Class Number",
                             "Enter the class number, in case you do not know the class number, press the Help button at the bottom of the main window and follow the instructions")


#Get Result Button
buttonRes = Button(root, text="RESULT", font="Raleway", bg="grey", fg="white", height=1, width=10, command=getResult)
buttonRes.grid(column=1, row=3)
buttonRes.place(x=180, y=75)






#Choose a Video
def open_file():
    global path
    browse_text.set("loading...")
    file = askopenfile(parent=root, mode='rb', title="Choose a Video", filetypes=[("Video file 1", "*.mp4"),("Video file 2", "*.webm")])
    if file:
        str = file.name
        index = str.rindex('/')
        str2 = str[index+1:]
        show = tk.Label(root, text=str2+"✔", font=("Raleway", 10), fg="green")
        show.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        show.place(x=25, y=111)
        path = file.name
    browse_text.set("Browse")






def main():
    root.mainloop()
    return path, classNumber

if __name__ == '__main__':
    main()



