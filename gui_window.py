import customtkinter
from tkinter import filedialog
import os
from read_lidar import LidarProcesser, State
# NEM!! import open3d as o3d NEM!!!
import subprocess
#import json
import numpy as np
from PIL import Image, ImageTk
import tkinter

class ButtonFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        # Define elements
        self.display_btn = customtkinter.CTkButton(self, text="Display", command=self.master.display_o3d, width=210, height=40)
        self.refresh_btn = customtkinter.CTkButton(self, text ="Refresh preview", command=self.master.refresh, width=210, height=40)
        self.reset_btn = customtkinter.CTkButton(self, text="Reset", command=self.master.reset, width=210, height=40)

        # Place elements
        self.display_btn.grid(row=0, column=0, padx=5, pady=5)
        self.refresh_btn.grid(row=0,column=1, padx=5, pady=5)
        self.reset_btn.grid(row=0,column=2, padx=5, pady=5)

class ParamFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.file_path = ()
        

        image_array = self.master.prc.render_image(self.master.prc.get_first_frame())
        
        # Convert the np array to PIL Image
        photo_img = Image.fromarray(image_array)
        photo = ImageTk.PhotoImage(photo_img)

        # Define elements
        self.optionmenu_var = customtkinter.StringVar(value="File")
        self.menu_bar = customtkinter.CTkOptionMenu(self,values=["Open", "Save point cloud"],
                                         command=self.__optionmenu_callback,
                                         variable=self.optionmenu_var)
        
        self.check_var_heatmap = customtkinter.StringVar(value="off_heatmap")
        self.checkbox_heatmap = customtkinter.CTkCheckBox(self, text="Danger detection", command=self.__heatmap_event,
                                     variable=self.check_var_heatmap, onvalue="on_heatmap", offvalue="off_heatmap")
        
        self.labal_range = customtkinter.CTkLabel(self, text="The lower and upper bounds to display\n(frame number):", fg_color="transparent", justify='left')
        self.entry_range_lower = customtkinter.CTkEntry(self, placeholder_text="Lower bound (included)", width=160)
        self.entry_range_upper = customtkinter.CTkEntry(self, placeholder_text="Upper bound (included)", width=160)

        self.check_var = customtkinter.StringVar(value="off_ground")
        self.checkbox_ground = customtkinter.CTkCheckBox(self, text="Filter ground",
                                     variable=self.check_var, onvalue="on_ground", offvalue="off_ground")
        self.entry_ground_tresh = customtkinter.CTkEntry(self, placeholder_text="Epsilon for treshold", width=160)
        self.entry_ground_iter = customtkinter.CTkEntry(self, placeholder_text="Number of iterations", width=160)
        
        self.label_clastering = customtkinter.CTkLabel(self, text="Chose a clustering method", fg_color="transparent")
        self.radio_var = tkinter.IntVar(value=0)
        self.radiobutton_clastering = customtkinter.CTkRadioButton(self, text="NONE",
                                             variable=self.radio_var, value=0)
        self.entry_eps = customtkinter.CTkEntry(self, placeholder_text="Epsilon", width=160)
        self.entry_min_samples = customtkinter.CTkEntry(self, placeholder_text="Minimum sample number", width=160)
        self.radiobutton_dbscan = customtkinter.CTkRadioButton(self, text="DBSCAN",
                                             variable=self.radio_var, value=1)
        self.radiobutton_kmeans = customtkinter.CTkRadioButton(self, text="K-MEANS",
                                             variable=self.radio_var, value=2)
        self.entry_n_init = customtkinter.CTkEntry(self, placeholder_text="Number of clusters", width=160)

        self.label_box = customtkinter.CTkLabel(self, text="Filter bounding boxes", fg_color="transparent")
        self.switch_box_var = customtkinter.StringVar(value="box_on")
        self.switch_box = customtkinter.CTkSwitch(self, text="Boundig box on",
                                 variable=self.switch_box_var, onvalue="box_on", offvalue="box_off")
        self.label_height = customtkinter.CTkLabel(self, text="Set maximum and minimum\nheight for bounding boxes", fg_color="transparent", justify='left')
        self.progressbar_max_h = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=25, variable=tkinter.IntVar(value=100))
        self.progressbar_min_h = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=25, variable=tkinter.IntVar(value=0))
        self.label_width = customtkinter.CTkLabel(self, text="Set maximum and minimum\nwidth for bounding boxes", fg_color="transparent", justify='left')
        self.progressbar_max_w = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=25, variable=tkinter.IntVar(value=100))
        self.progressbar_min_w = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=25, variable=tkinter.IntVar(value=0))
        self.label_length = customtkinter.CTkLabel(self, text="Set maximum and minimum\nlength for bounding boxes", fg_color="transparent", justify='left')
        self.progressbar_max_l = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=25, variable=tkinter.IntVar(value=100))
        self.progressbar_min_l = customtkinter.CTkSlider(self, from_=0, to=100, number_of_steps=25, variable=tkinter.IntVar(value=0))

        self.img_label = tkinter.Label(image=photo) # TODO: átrakni a masterbe
        self.img_label.image = photo

        # Place elements
        self.menu_bar.grid(row=0, column=0, padx=20, pady=(10, 20), sticky="w")

        self.checkbox_heatmap.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="w")

        self.labal_range.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        self.entry_range_lower.grid(row=3, column=0, padx=20, pady=(5, 5), sticky="w")
        self.entry_range_upper.grid(row=4, column=0, padx=20, pady=(5, 20), sticky="w")

        self.checkbox_ground.grid(row=5, column=0, padx=20, pady=(10, 5), sticky="w")
        self.entry_ground_tresh.grid(row=6, column=0, padx=20, pady=(5, 5), sticky="w")
        self.entry_ground_iter.grid(row=7, column=0, padx=20, pady=(5, 20), sticky="w")

        self.label_clastering.grid(row=8, column=0, padx=20, pady=(10, 5), sticky="w")
        self.radiobutton_clastering.grid(row=9, column=0, padx=20, pady=(5, 10), sticky="w")
        self.radiobutton_dbscan.grid(row=10, column=0, padx=20, pady=(10, 5), sticky="w")
        self.entry_eps.grid(row=11, column=0, padx=20, pady=(5, 5), sticky="w")
        self.entry_min_samples.grid(row=12, column=0, padx=20, pady=(5, 10), sticky="w")
        self.radiobutton_kmeans.grid(row=13, column=0, padx=20, pady=(10, 5), sticky="w")
        self.entry_n_init.grid(row=14, column=0, padx=20, pady=(5, 20), sticky="w")

        self.label_box.grid(row=15, column=0, padx=20, pady=(10, 5), sticky="w")
        self.switch_box.grid(row=16, column=0, padx=20, pady=(5, 10), sticky="w")
        self.label_height.grid(row=17, column=0, padx=20, pady=(5, 5), sticky="w")
        self.progressbar_max_h.grid(row=18, column=0, padx=20, pady=(5, 5), sticky="w")
        self.progressbar_min_h.grid(row=19, column=0, padx=20, pady=(5, 10), sticky="w")
        self.label_width.grid(row=20, column=0, padx=20, pady=(5, 5), sticky="w")
        self.progressbar_max_w.grid(row=21, column=0, padx=20, pady=(5, 5), sticky="w")
        self.progressbar_min_w.grid(row=22, column=0, padx=20, pady=(5, 10), sticky="w")
        self.label_length.grid(row=23, column=0, padx=20, pady=(5, 5), sticky="w")
        self.progressbar_max_l.grid(row=24, column=0, padx=20, pady=(5, 5), sticky="w")
        self.progressbar_min_l.grid(row=25, column=0, padx=20, pady=(5, 10), sticky="w")
        

        self.img_label.grid(row=0, column=1, padx=20, pady=15)

    def __heatmap_event(self):
        if self.check_var_heatmap.get() == 'on_heatmap':
            state_var = "readonly"
        else:
            state_var = "normal"

        self.entry_range_lower.configure(state=state_var)
        self.entry_range_upper.configure(state=state_var)
        self.checkbox_ground.configure(state=state_var)
        self.entry_ground_tresh.configure(state=state_var)
        self.entry_ground_iter.configure(state=state_var)
        self.radiobutton_clastering.configure(state=state_var)
        self.entry_eps.configure(state=state_var)
        self.entry_min_samples.configure(state=state_var)
        self.radiobutton_dbscan.configure(state=state_var)
        self.radiobutton_kmeans.configure(state=state_var)
        self.entry_n_init.configure(state=state_var)
        self.switch_box.configure(state=state_var)
        self.progressbar_max_h.configure(state=state_var)
        self.progressbar_min_h.configure(state=state_var)
        self.progressbar_max_w.configure(state=state_var)
        self.progressbar_min_w.configure(state=state_var)
        self.progressbar_max_l.configure(state=state_var)
        self.progressbar_min_l.configure(state=state_var)
    
    def clear_entry(self):
        self.entry_range_lower.delete(0,100)
        self.entry_range_upper.delete(0,100)
        self.entry_ground_tresh.delete(0,100)
        self.entry_ground_iter.delete(0,100)
        self.entry_eps.delete(0,100)
        self.entry_min_samples.delete(0,100)
        self.entry_n_init.delete(0,100)

        # reconf placeholder because it sometimes disappears
        self.entry_range_lower.configure(placeholder_text='Lower bound (included)')
        self.entry_range_upper.configure(placeholder_text='Upper bound (included)')
        self.entry_ground_tresh.configure(placeholder_text='Epsilon for treshold')
        self.entry_ground_iter.configure(placeholder_text='Number of iterations')
        self.entry_eps.configure(placeholder_text='Epsilon')
        self.entry_min_samples.configure(placeholder_text='Minimum sample number')
        self.entry_n_init.configure(placeholder_text='Number of clusters')
    
    def conf_progressbar(self):
        if self.master.prc.is_empty():
            width, length, height = 100, 100, 100
        else:
            width, length, height = self.master.prc.get_extents()
        self.progressbar_max_h.configure(to=height)
        self.progressbar_min_h.configure(to=height)
        self.progressbar_max_w.configure(to=width)
        self.progressbar_min_w.configure(to=width)
        self.progressbar_max_l.configure(to=length)
        self.progressbar_min_l.configure(to=length)

    def set_default_progressbar(self):
        if self.master.prc.is_empty():
            width, length, height = 100, 100, 100
        else:
            width, length, height = self.master.prc.get_extents()
        self.progressbar_max_h.set(height)
        self.progressbar_min_h.set(0)
        self.progressbar_max_w.set(width)
        self.progressbar_min_w.set(0)
        self.progressbar_max_l.set(length)
        self.progressbar_min_l.set(0)
        

    # File management
    def __optionmenu_callback(self, choice):
        if choice == "Open":
            self.__select_file()

            if not self.file_path:
                return

            # Refresh parameters
            if not self.master.prc.is_empty():
                self.labal_range.configure(text=f"The lower and upper bounds to display\n({1} - {self.master.prc.get_frame_num()}):")
                self.conf_progressbar()
                self.checkbox_ground.deselect()
                self.checkbox_heatmap.deselect()

                self.refresh_preview()
            else:
                dialog = customtkinter.CTk()
                dialog.geometry("300x100")
                dialog.title("WARNING")
                warning = customtkinter.CTkLabel(dialog, text="Something went wrong with reading the file!\nCheck the file!")
                warning.pack()
                dialog.mainloop()
        else:
            if not self.file_path:
                dialog = customtkinter.CTk()
                dialog.geometry("300x100")
                dialog.title("WARNING")
                warning = customtkinter.CTkLabel(dialog, text="Please open a file first!")
                warning.pack()
                dialog.mainloop()
            else:
                self.__save_pcd()

        print("optionmenu dropdown clicked:", choice)

    def __save_pcd(self):
        file_name = filedialog.asksaveasfilename(defaultextension='.pcd', filetypes=[("point cloud", '*.pcd')])
        self.master.prc.save(file_name, self.get_lower())

    # File selection
    def __select_file(self):
        self.file_path = filedialog.askopenfilename() # if nothing is selected then it's an empy tuple

        if not self.file_path:
            return

        _, ext = os.path.splitext(self.file_path)

        if ext not in LidarProcesser.ValidExtensions:
            return
        
        self.master.prc.new_file(self.file_path)

        print(self.file_path)

    def refresh_preview(self):
        # Check if there is any file loaded
        if self.master.prc.is_empty():
            return
        
        image = Image.fromarray(self.master.prc.render_image(self.master.prc.get_first_frame()))
        
        # Create a PhotoImage from the PIL Image
        photo = ImageTk.PhotoImage(image)
        
        # Update the image on the label
        self.img_label.config(image=photo)
        self.img_label.image = photo  # Store a reference

        if self.master.prc.get_status() == State.DEFAULT or self.master.prc.get_status() == State.GROUND_FILTER:
            # Refresh details
            text_frame = f"Number of frames: {self.master.prc.get_frame_num()}"
            text_avg_size = f"Average point cloud size: {self.master.prc.get_avg_pcd_len()}"
            text_cluster_num = f"Average number of clusters: {self.master.prc.get_cluster_num()}"
            text_box_num = f"Average number of bounding boxes: {self.master.prc.get_box_num()}"
            text_time = f"Time spent with clustering: {self.master.prc.get_cluster_time()} s"
            self.master.info_label.configure(text='\n'.join([
                text_frame + "\t\t\t" + text_avg_size, 
                text_cluster_num + "\t\t" + text_box_num,
                text_time
            ]))

    def get_lower(self):
        # check value
        val = self.entry_range_lower.get()
        max_val = self.master.prc.get_frame_num()
        if not val.isdigit() or int(val)-1 < 0:
            val = 0
        elif int(val) > max_val:
            val = max_val
        else:
            val = int(val) -1

        return val

    def get_upper(self):
        # check value
        val = self.entry_range_upper.get()
        max_val = self.master.prc.get_frame_num()
        if not val.isdigit() or int(val) > max_val:
            val = max_val
        else:
            val = int(val)

        return val
    
    def get_min_samples(self):
        val = self.entry_min_samples.get()
        if not val.isdigit() or val == '0':
            val = 1
        else:
            val = int(val)

        return val
    
    def get_eps(self):
        val = self.entry_eps.get()
        if not val.replace('.','',1).isdigit() or float(val) == 0:
            val = 0.1
        else:
            val = float(val)

        return val

    def get_n_init(self):
        val = self.entry_n_init.get()
        if not val.isdigit() or val == '0':
            val = 1
        else:
            val = int(val)

        return val
    
    def get_iter_num(self):
        val = self.entry_ground_iter.get()
        if not val.isdigit() or val == '0':
            val = 1
        else:
            val = int(val)

        return val
    
    def get_treshold(self):
        val = self.entry_ground_tresh.get()
        if not val.replace('.','',1).isdigit():
            val = 0.0001
        else:
            val = float(val)

        return val

    
    def filter_ground(self):
        lower = self.get_lower()
        upper = self.get_upper()
        treshold = self.get_treshold()
        iter = self.get_iter_num()

        self.master.prc.filter_ground(lower, upper, iter, treshold)

    def clustering_colors(self):
        extents = self.__get_filters()
        val = self.radio_var.get()
        lower = self.get_lower()
        upper = self.get_upper()

        if val == 1:
            self.master.prc.dbscan_colors(self.get_eps(), self.get_min_samples(), lower, upper, extents)
        elif val == 2:
            self.master.prc.kmeans_colors(self.get_n_init(), lower, upper, extents)
        elif val == 0:
            self.master.prc.none_colors(lower, upper)
    
    def __get_filters(self):
        max_w = int(self.progressbar_max_w.get())
        min_w = int(self.progressbar_min_w.get())
        max_h = int(self.progressbar_max_h.get())
        min_h = int(self.progressbar_min_h.get())
        max_l = int(self.progressbar_max_l.get())
        min_l = int(self.progressbar_min_l.get())

        if max_w < min_w:
            max_w, min_w = min_w, max_w

        if max_h < min_h:
            max_h, min_h = min_h, max_h
        
        if max_l < min_l:
            max_l, min_l = min_l, max_l

        return max_w, min_w, max_h, min_h, max_l, min_l

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.prc = LidarProcesser()

        # Init parameters
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("green") 
        self.title("Interactive LiDAR Playground")
        self.geometry("980x640")
        self.minsize(980,640)
        self.maxsize(980,640)
        self.grid_columnconfigure((0, 1), weight=1)
        
        # Define elements
        self.parameter_frame = ParamFrame(self, width=250, height=600)
        self.button_frame = ButtonFrame(self)
        self.info_label = customtkinter.CTkLabel(self, text="Details...", fg_color="transparent", justify='left')      

        # Place elements
        self.parameter_frame.grid(row=0, rowspan=3, column=0, padx=10, pady=(10, 0), sticky="nsw")
        self.info_label.grid(row=1, column=1, padx=20, pady=(0, 0), sticky="nsw")
        self.button_frame.grid(row=2, column=1, padx=10, pady=(10, 0), sticky="nsw")
        

    def display_o3d(self):
        if self.prc.is_empty():
            dialog = customtkinter.CTk()
            dialog.geometry("300x100")
            dialog.title("WARNING")
            warning = customtkinter.CTkLabel(dialog, text="Please open a file first!")
            warning.pack()
            dialog.mainloop()
            return
        
        self.refresh()

        if not self.prc.is_pcap() and self.prc.get_status() == State.HEATMAP:
            return

        lower = self.parameter_frame.get_lower()
        upper = self.parameter_frame.get_upper()

        if not self.prc.is_show_box():
            line_point_list = []
            line_line_list = []
        else:
            line_point_list = self.prc.get_lineset_p()[lower:upper]
            line_line_list = self.prc.get_lineset_l()[lower:upper]
        
        point_array = self.prc.get_array(lower, upper)
        color_array = self.prc.get_color_array(lower, upper)
        


        """
        shape = point_array.shape
        byte_string = point_array.tobytes()
        encoded_string = base64.b64encode(byte_string).decode()
        # ARGUMENT LIST TOO LONG
        """
        
        """
        serialized_array = json.dumps(point_array.tolist())

        with open('visualize.py', 'w') as file:
            file.write(
        f'''import open3d as o3d
        import json
        point_array = json.loads("{serialized_array}")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_array)
        o3d.visualization.draw_geometries([point_cloud])'''
        )

        process = subprocess.Popen(['python3', 'visualize.py'])
        process.wait()

        # LEHET EZ LESZ MÉGIS? Tabokat kitörölni!!
        """

        np.save('temp_p.npy', point_array)
        np.save('temp_c.npy', color_array)
        np.save('temp_lp.npy', line_point_list)
        np.save('temp_ll.npy', line_line_list)


        process = subprocess.Popen(['python3', 'visualizer.py'])
        process.wait()
    
    def refresh(self):
        if self.prc.is_empty():
            dialog = customtkinter.CTk()
            dialog.geometry("300x100")
            dialog.title("WARNING")
            warning = customtkinter.CTkLabel(dialog, text="Please open a file first!")
            warning.pack()
            dialog.mainloop()
            return
        
        if self.parameter_frame.check_var_heatmap.get() == 'on_heatmap':
            if not self.prc.is_pcap():
                dialog = customtkinter.CTk()
                dialog.geometry("400x100")
                dialog.title("WARNING")
                warning = customtkinter.CTkLabel(dialog, text="Danger detection function is only available with pcap file format!")
                warning.pack()
                dialog.mainloop()
                return
            
            self.prc.set_status(State.HEATMAP)
            self.prc.set_show_box(False)

            self.prc.heatmap_colors(900)
        else:
            if self.parameter_frame.check_var.get() == 'on_ground':
                self.prc.set_status(State.GROUND_FILTER)

                self.parameter_frame.filter_ground()
            else:
                self.prc.set_status(State.DEFAULT)

            if self.parameter_frame.switch_box_var.get() == 'box_on':
                self.prc.set_show_box(True)
            else:
                self.prc.set_show_box(False)

            self.parameter_frame.clustering_colors()
        
        self.parameter_frame.refresh_preview()

    def reset(self):
        self.parameter_frame.checkbox_ground.deselect()
        self.parameter_frame.checkbox_heatmap.deselect()
        self.parameter_frame.radiobutton_clastering.select(0)
        self.parameter_frame.switch_box.select("box_off")
        self.parameter_frame.clear_entry()
        self.parameter_frame.conf_progressbar()
        self.parameter_frame.set_default_progressbar()
        if not self.prc.is_empty():
            self.prc.set_status(State.DEFAULT)
        self.prc.set_show_box(True)

