from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
from core_classify import *
import pandas as pd
import time
import threading
import sys


class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self._frame = None
        self.switch_frame(LoginPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

    def repack(self):
        self._frame.pack()


class LoginPage(Frame):
    def __init__(self, master: App):
        Frame.__init__(self, master)
        self.width = 450
        self.height = 200
        self.passcode = "123456"

        self.master.title("Please enter the passcode")
        self.master.geometry(f"{self.width}x{self.height}")
        self.configure(width=self.width, height=self.height)

        # entry label
        self.code_label = Label(self, text='Code')
        self.code_label.place(relx=0.3, rely=0.3, anchor="n")

        # entry
        self.code_var = StringVar()
        self.code_entry = Entry(self, textvariable=self.code_var)
        self.code_entry.place(relx=0.6, rely=0.3, anchor="n")

        # login btn
        self.login_btn = Button(self, text="Verify code", command=self.btn_login)
        self.login_btn.place(relx=0.5, rely=0.7, anchor="center")

    def btn_login(self):
        if self.passcode == self.code_var.get():
            self.master.switch_frame(MainApp)
        else:
            tkinter.messagebox.showerror(title="Error", message="Passcode wrong.")


class MainApp(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.filename = None
        self.width = 400
        self.height = 500
        self.all_buttons = []
        self.max_no_action_time = 30 * 60
        self.no_action_start = time.time()
        self.timer_stop_flag = False
        self.exit_thread_flag = FALSE

        self.master.protocol('WM_DELETE_WINDOW', self.exit)

        self.timer = threading.Thread(target=self.no_action_thread)
        self.timer.start()

        self.master.title("Spam email classifier")
        self.master.geometry(f"{self.width}x{self.height}")
        self.configure(width=self.width, height=self.height)

        # select input file btn
        self.select_file_btn = Button(self, text="Choose a csv file to classify", command=self.btn_file_selection)
        self.select_file_btn.place(relx=0.5, rely=0.5, anchor=CENTER)

        # begin classify btn
        self.begin_classify_btn = Button(self, text="Begin", command=self.btn_begin_classify, state="disable")
        self.begin_classify_btn.place(relx=0.5, rely=0.65, anchor=CENTER)

        # info label
        self.info_label = Label(self, text="Please select a data file.")
        self.info_label.place(relx=0.5, rely=0.35, anchor=CENTER)

        self.file_info_label = Label(self, text="")
        self.file_info_label.place(relx=0.5, rely=0.40, anchor=CENTER)

        # all button list
        self.all_buttons.append(self.select_file_btn)
        self.all_buttons.append(self.begin_classify_btn)

        self.reset_timer()

    def exit(self):
        self.exit_thread_flag = True
        self.master.destroy()
        exit()

    def no_action_thread(self):
        self.no_action_start = time.time()
        while (time.time() - self.no_action_start) < self.max_no_action_time:
            if self.exit_thread_flag:
                exit()
            if self.timer_stop_flag:
                self.no_action_start = time.time()
            print(f"{time.time() - self.no_action_start}<{self.max_no_action_time}")
            time.sleep(0.1)

        self.master.switch_frame(LoginPage)

    def reset_timer(self):
        self.no_action_start = time.time()
        self.start_timer()

    def stop_timer(self):
        self.timer_stop_flag = True

    def start_timer(self):
        self.timer_stop_flag = False

    def reset_states(self):
        self.filename = None
        self.info_label.configure(text="Please select a data file.")
        self.begin_classify_btn.configure(state="disable")
        self.select_file_btn.configure(state="normal")
        self.file_info_label.configure(text="")
        self.reset_timer()

    def unfreeze_all_buttons(self):
        for btn in self.all_buttons:
            btn: Button
            btn.configure(state="normal")

    def freeze_all_buttons(self):
        for btn in self.all_buttons:
            btn: Button
            btn.configure(state="disable")

    def btn_file_selection(self):
        self.reset_timer()

        input_file = tkinter.filedialog.askopenfilename()
        if input_file != "":
            self.filename = input_file
            self.info_label.configure(text=f"Data:{input_file}")
            self.begin_classify_btn.configure(state="normal")
            record_num = pd.read_csv(self.filename).shape[0]
            self.file_info_label.configure(text=f"Found record:{record_num}")

    def btn_begin_classify(self):
        cls_df: pd.DataFrame
        cls_df = None

        self.reset_timer()
        self.stop_timer()

        start_time = None
        time_elapsed = None

        try:
            self.info_label.configure(text="Please wait, system processing...")
            self.freeze_all_buttons()
            self.update()  # force update the screen
            start_time = time.time()
            cls_df = classify_spam(self.filename)
            time_elapsed = time.time() - start_time
            self.unfreeze_all_buttons()
        except Exception:
            tkinter.messagebox.showerror(title="Error", message="Some error ocurred")
            self.reset_states()

        if cls_df is not None:
            tkinter.messagebox.showinfo(title="Success", message=f"Result file successfully generated.\nTime used :{time_elapsed:.2f} seconds.")
            output_path = tkinter.filedialog.asksaveasfilename(filetypes=[('CSV file', '.csv')])
            if output_path != "":
                cls_df.to_csv(output_path, index=FALSE)
            else:
                tkinter.messagebox.showerror(title="Error", message="User abort, file not saved.")
            self.reset_states()
        else:
            tkinter.messagebox.showerror(title="Error", message="Result is empty.")
            self.reset_states()


if __name__ == '__main__':
    app = App()
    app.mainloop()
