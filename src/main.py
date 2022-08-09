import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import threading
import time



class Interface(tk.Tk):
    def __init__(self):
        """初始化介面"""
        super().__init__()  # 有點相當於tk.Tk()

        self.title('FS8 Image Simulator')
        self.minsize(572, 687)  # w, h
        self.iconbitmap('../asset/satelliteIcon.ico')
        self.config(bg='#2B2B2B')

        """frame 1"""
        self.entryvarDEM = tk.StringVar()
        self.entryvarTif = tk.StringVar()
        self.entryvarEph = tk.StringVar()
        self.entryvarOut = tk.StringVar()
        self.entryvarmp4 = tk.StringVar()
        self.entryvarTgLat = tk.StringVar()
        self.entryvarTgLon = tk.StringVar()
        self.entryvarsigma_O = tk.StringVar(value=0)
        self.entryvarsigma_P = tk.StringVar(value=0)
        self.entryvarsigma_K = tk.StringVar(value=0)

        # self.entryvarTgLat.trace_add('write', self.saveLat_change())
        # self.entryvarTgLon.trace_add('write', self.saveLon_change())

        # frame width 約為572
        frame1 = tk.Frame(self, bg='#2B2B2B')
        frame1.pack(side=tk.TOP)

        label1 = tk.Label(frame1, text='1. Load the following files:', font='Helvetica 10 bold', bg='#2B2B2B', fg='#F3F4F4', padx=10, pady=5)

        labelDEM = tk.Label(frame1, text='a.  Load DEM file ( .asc ):', bg='#2B2B2B', fg='#F3F4F4', width='20', anchor='w', padx=10, pady=5)
        labelTif = tk.Label(frame1, text='b.  Load Ortho Image ( .tif ):', bg='#2B2B2B', fg='#F3F4F4', width='20', anchor='w', padx=10, pady=5)
        labelEph = tk.Label(frame1, text='c.  Load Ephemeris ( .dim ):', bg='#2B2B2B', fg='#F3F4F4', width='20', anchor='w', padx=10, pady=5)
        # width 設定label寬度 / anchor='w'讓文字靠左對齊 / padx=30, , pady=5, 設定label離邊框距離

        entryDEM = tk.Entry(frame1, textvariable=self.entryvarDEM, width='45', bg='#595959', fg='#F3F4F4')
        entryTif = tk.Entry(frame1, textvariable=self.entryvarTif, width='45', bg='#595959', fg='#F3F4F4')
        entryEph = tk.Entry(frame1, textvariable=self.entryvarEph, width='45', bg='#595959', fg='#F3F4F4')

        buttonDEM = tk.Button(frame1, text="Browse ", command=self.opendirDEM, bg='#FFA500', height="1")
        buttonTif = tk.Button(frame1, text="Browse ", command=self.opendirTif, bg='#FFA500',  height="1")
        buttonEph = tk.Button(frame1, text="Browse ", command=self.opendirEph, bg='#FFA500',  height="1")

        # 放置位置
        label1.grid(row=0, columnspan=3, sticky='W', padx=5, pady=5)
        labelDEM.grid(row=1, column=0, padx=(20, 5))
        labelTif.grid(row=2, column=0, padx=(20, 5))
        labelEph.grid(row=3, column=0, padx=(20, 5))

        entryDEM.grid(row=1, column=1, padx=0)
        entryTif.grid(row=2, column=1, padx=0)
        entryEph.grid(row=3, column=1, padx=0)

        buttonDEM.grid(row=1, column=2, padx=5, pady=5, sticky=tk.E)
        buttonTif.grid(row=2, column=2, padx=5, pady=5, sticky=tk.E)
        buttonEph.grid(row=3, column=2, padx=5, pady=5, sticky=tk.E)

        """frame 2"""
        frame2 = tk.Frame(self, height=80, bg='#2B2B2B')
        frame2.pack(side=tk.TOP)

        label2 = tk.Label(frame2, text='2. Input target position in degrees:', font='Helvetica 10 bold',
                          bg='#2B2B2B', fg='#F3F4F4', padx=10, pady=5)
        label2.grid(row=0, columnspan=3, sticky='W', padx=(5, 162), pady=5)

        labelTgLat = tk.Label(frame2, text='Latitude:', bg='#2B2B2B', fg='#F3F4F4', anchor='w', padx=10, pady=5)
        labelTgLon = tk.Label(frame2, text='Longitude:', bg='#2B2B2B', fg='#F3F4F4', anchor='w', padx=10, pady=5)
        entryTgLat = tk.Entry(frame2, textvariable=self.entryvarTgLat, width="20", bg='#595959', fg='#F3F4F4')
        entryTgLon = tk.Entry(frame2, textvariable=self.entryvarTgLon, width="20", bg='#595959', fg='#F3F4F4')
        labelTgLat.grid(row=1, column=0, sticky='W', padx=(20, 8), pady=5)
        labelTgLon.grid(row=1, column=2, sticky='W', padx=(20, 8), pady=5)
        entryTgLat.grid(row=1, column=1, sticky='E', padx=10, pady=5)
        entryTgLon.grid(row=1, column=3, sticky='E', padx=10, pady=5)

        """frame 3"""
        frame3 = tk.Frame(self, width=572, bg='#2B2B2B')
        frame3.pack(side=tk.TOP)
        label3 = tk.Label(frame3, text='3. Select an output file:', font='Helvetica 10 bold',
                          bg='#2B2B2B', fg='#F3F4F4', padx=10, pady=5)
        label3.grid(row=0, columnspan=3, sticky='W', padx=5, pady=5)

        labelOut = tk.Label(frame3, text='Output directory :', bg='#2B2B2B', fg='#F3F4F4', width='20', anchor='w', padx=10, pady=5)
        entryOut = tk.Entry(frame3, textvariable=self.entryvarOut, width='45', bg='#595959', fg='#F3F4F4')
        buttonOut = tk.Button(frame3, text="Browse ", command=self.opendirOut, bg='#FFA500', height="1")
        labelOut.grid(row=1, column=0, padx=(20, 5))
        entryOut.grid(row=1, column=1, padx=0)
        buttonOut.grid(row=1, column=2, padx=5, pady=5, sticky=tk.E)

        """frame 4"""
        frame4 = tk.Frame(self, width=572, bg='#2B2B2B')
        frame4.pack(side=tk.TOP)
        label4 = tk.Label(frame4,
                          text='4. ( Optional ) Add noise to the orientation angle in degrees (default sigma = 0):'
                          , font='Helvetica 10 bold', bg='#2B2B2B', fg='#F3F4F4', padx=10, pady=5)
        label4.grid(row=0, columnspan=6, sticky='W', padx=5, pady=5)

        labelO = tk.Label(frame4, text='Omega:', bg='#2B2B2B', fg='#F3F4F4', anchor='w',
                          padx=10, pady=5)
        labelP = tk.Label(frame4, text='Phi:', bg='#2B2B2B', fg='#F3F4F4', anchor='w',
                          padx=10, pady=5)
        labelK = tk.Label(frame4, text='Kappa:', bg='#2B2B2B', fg='#F3F4F4', anchor='w',
                          padx=10, pady=5)
        # width 設定label寬度 / anchor='w'讓文字靠左對齊 / padx=30, , pady=5, 設定label離邊框距離

        entryO = tk.Entry(frame4, textvariable=self.entryvarsigma_O, width='8', bg='#595959', fg='#F3F4F4')
        entryP = tk.Entry(frame4, textvariable=self.entryvarsigma_P, width='8', bg='#595959', fg='#F3F4F4')
        entryK = tk.Entry(frame4, textvariable=self.entryvarsigma_K, width='8', bg='#595959', fg='#F3F4F4')

        labelO.grid(row=1, column=0, padx=(20, 0))
        labelP.grid(row=1, column=2, padx=0)
        labelK.grid(row=1, column=4, padx=0)

        entryO.grid(row=1, column=1, padx=(5, 20))
        entryP.grid(row=1, column=3, padx=(5, 20))
        entryK.grid(row=1, column=5, padx=(5, 139))

        self.buttonlaunch = tk.Button(frame4, text="Launch ", command=lambda: self.btnlaunch(),
                                      bg='#2488C0', font='Helvetica 10 bold', pady=2)
        self.buttonlaunch.grid(row=2, columnspan=6, padx=5, pady=5, sticky=tk.W + tk.E)

        """bottom frame"""
        bottomframe = tk.Frame(self, height=80, width=572, bg='#2B2B2B')
        bottomframe.pack(side=tk.TOP)

        self.text = tk.Text(bottomframe, bg='#595959', fg='#F3F4F4', height=20, width=77)
        self.text.grid(row=0, sticky="nsew", padx=(6, 0), pady=5)
        self.text.tag_configure("stderr", foreground="#b22222")

        # show logs in the text widget
        sys.stdout = StdRedirector(self.text)
        sys.stderr = StdRedirector(self.text)

        scrollbar = tk.Scrollbar(bottomframe, command=self.text.yview)
        scrollbar.grid(row=0, column=1, sticky='nsew', padx=(0, 6), pady=5)
        self.text['yscrollcommand'] = scrollbar.set

    def opendirDEM(self):
        self.dirname = filedialog.askopenfilename(initialdir="/",
                                                     title="Select DEM",
                                                     filetype=(("asc files", "*.asc"), ("all files", "*.*")))
        self.entryvarDEM.set(self.dirname)  # 設定變數entryvar，等同於設定部件Entry

        if not self.dirname:
            messagebox.showwarning('Warning', message='File not selected！')  # 彈出訊息提示框
        if self.dirname and not self.dirname.endswith('.asc'):
            messagebox.showwarning('Warning', message='Select .asc files only！')  # 彈出訊息提示框

    def opendirTif(self):
        self.dirname = filedialog.askopenfilename(initialdir="/",
                                                     title="Select Tif",
                                                     filetype=(("Tif files", "*.tif"), ("all files", "*.*")))
        self.entryvarTif.set(self.dirname)  # 設定變數entryvar，等同於設定部件Entry
        if not self.dirname:
            messagebox.showwarning('Warning', message='File not selected！')  # 彈出訊息提示框
        if self.dirname and not self.dirname.endswith('.tif'):
            messagebox.showwarning('Warning', message='Select .tif files only！')  # 彈出訊息提示框

    def opendirEph(self):
        self.dirname = filedialog.askopenfilename(initialdir="/",
                                                     title="Select ephemeris data",
                                                     filetype=(("Eph files", "*.dim"), ("all files", "*.*")))
        self.entryvarEph.set(self.dirname)  # 設定變數entryvar，等同於設定部件Entry
        if not self.dirname:
            messagebox.showwarning('Warning', message='File not selected！')  # 彈出訊息提示框
        if self.dirname and not self.dirname.endswith('.dim'):
            messagebox.showwarning('Warning', message='Select .dim files only！')  # 彈出訊息提示框

    def opendirOut(self):
        self.dirname = filedialog.askdirectory(initialdir="/",
                                                  title="Select output file")
        self.entryvarOut.set(self.dirname)  # 設定變數entryvar，等同於設定部件Entry
        if not self.dirname:
            messagebox.showwarning('警告', message='File not selected！')  # 彈出訊息提示框

    def btnlaunch(self):

        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)

        if self.entryvarDEM.get().endswith('.asc') and len(self.entryvarTgLat.get()) != 0 \
                and len(self.entryvarTgLon.get()) != 0:
            from loadData import DEM
            self.DEM = DEM(self.entryvarDEM.get())
            dem = self.DEM.loadDem()
            self.checkLatLon()

        if self.entryvarDEM.get().endswith('.asc') and self.entryvarTif.get().endswith('.tif') \
                and self.entryvarEph.get().endswith('.dim') \
                and len(self.entryvarTgLat.get()) != 0 \
                and len(self.entryvarTgLon.get()) != 0 \
                and len(self.entryvarOut.get()) != 0 \
                and len(self.entryvarsigma_O.get()) != 0 \
                and len(self.entryvarsigma_P.get()) != 0 \
                and len(self.entryvarsigma_K.get()) != 0:
            msgbox = messagebox.askquestion('Reminders', 'Simulation.mp4 and metadata.txt will be saved '
                                               'in the selected output directory.',
                                               icon='info')
            if msgbox == 'yes':
                self.run_Topdown()

        elif len(self.entryvarTgLat.get()) == 0 or len(self.entryvarTgLon.get()) == 0:
            messagebox.showerror('Error', 'Please input target position !')

        elif len(self.entryvarsigma_O.get()) == 0 or len(self.entryvarsigma_P.get()) == 0 \
                or len(self.entryvarsigma_K.get()) == 0:
            messagebox.showerror('Error',
                                    'The standard deviation of the Gaussian distribution is missing !')

        else:
            messagebox.showerror('Error', 'File Missing or Incorrect !')

    def run_Topdown(self):
        from loadData import DEM, Tif, Eph
        from topDown_input import TopDown
        DEM = DEM(self.entryvarDEM.get())
        Tif = Tif(self.entryvarTif.get())
        Eph = Eph(self.entryvarEph.get())
        t = TopDown([4.5 * 10 ** (-6), 2560, 2560], [0, 0, 3927 * 10 ** (-3)], DEM, Tif, Eph)

        thread1 = threading.Thread(target=t.colinearityEquation(
                                                            fps=12,
                                                            outputdir=self.entryvarOut.get(),
                                                            targetLat=self.entryvarTgLat.get(),
                                                            targetLon=self.entryvarTgLon.get(),
                                                            sigma_O=float(self.entryvarsigma_O.get()),
                                                            sigma_P=float(self.entryvarsigma_P.get()),
                                                            sigma_K=float(self.entryvarsigma_K.get())), args=(5,))
        thread1.daemon = True  # close pipe if GUI process exits
        thread1.start()

        from utilities import image2gif
        loading_process = threading.Thread(name='process',
                                           target=image2gif(outputdir=self.entryvarOut.get(), fps=12))
        loading_process.daemon = True
        loading_process.start()
        while loading_process.is_alive():
            self.animate_loading()

    def animate_loading(self):      # loading animation
        chars = "/—\|"
        for char in chars:
            sys.stdout.write('\r' + 'loading...' + char)
            time.sleep(.1)
            sys.stdout.flush()

    def checkLatLon(self):
        from coordinateSystem import CoordinateSystem
        cs = CoordinateSystem()
        xlucorner = self.DEM.__dict__['xlucorner']
        ylucorner = self.DEM.__dict__['ylucorner']
        numcol = self.DEM.__dict__['cols']
        numrow = self.DEM.__dict__['rows']
        cellsize = self.DEM.__dict__['cellsize']

        x_min = xlucorner
        y_max = ylucorner
        x_max = x_min + numcol * cellsize
        y_min = y_max - numrow * cellsize

        targetX, targetY = cs.LatLon_To_TWD97TM2(float(self.entryvarTgLat.get()), float(self.entryvarTgLon.get()))

        if not (x_min <= targetX <= x_max) or not (y_min <= targetY <= y_max):
            messagebox.showerror("Error", 'target point not within DEM, please select another target !')

# redirect output to GUi
class StdRedirector():
    """Class that redirects the stdout and stderr to the GUI console"""
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        """Updates the console widget with the stdout and stderr output"""
        self.text_space.config(state=tk.NORMAL)
        self.text_space.insert("end", string)
        self.text_space.see("end")
        # self.text_space.update_idletasks()
        self.text_space.update()
        self.text_space.config(state=tk.DISABLED)
        # 若要避免程式停止回應，要用queue的方式寫
        # cf: https://stackoverflow.com/questions/55796812/tkinter-is-sometimes-freezing-when-i-call-update-idletasks
        # cf: https://stackoverflow.com/questions/16745507/tkinter-how-to-use-threads-to
        #     -preventing-main-event-loop-from-freezing/16747734#16747734

    def flush(self):
        pass


if __name__ == '__main__':
    interface = Interface()
    interface.mainloop()
