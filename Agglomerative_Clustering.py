import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk, Frame
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import os

class AgglomerativeClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation with Agglomerative Clustering")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.original_img = None
        self.processed_img = None
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Control panel (left side)
        control_panel = Frame(main_frame, bg="#f0f0f0", width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # App title
        title_label = tk.Label(control_panel, text="Agglomerative\nClustering", 
                             font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333333")
        title_label.pack(pady=(0, 20))
        
        # Upload button
        upload_frame = Frame(control_panel, bg="#f0f0f0")
        upload_frame.pack(fill=tk.X, pady=10)
        
        self.upload_btn = ttk.Button(
            upload_frame, 
            text="Upload Image",
            command=self.load_image,
            style="Accent.TButton"
        )
        self.upload_btn.pack(fill=tk.X, ipady=8)
        
        # File info
        self.file_info = tk.Label(control_panel, text="No file selected", 
                                font=("Arial", 10), fg="#555555", bg="#f0f0f0",
                                wraplength=250)
        self.file_info.pack(fill=tk.X, pady=5)
        
        # Cluster slider
        slider_frame = Frame(control_panel, bg="#f0f0f0")
        slider_frame.pack(fill=tk.X, pady=20)
        
        slider_label = tk.Label(slider_frame, text="Number of Clusters:", 
                              font=("Arial", 12), bg="#f0f0f0")
        slider_label.pack(anchor=tk.W)
        
        self.clusters_var = tk.IntVar(value=4)
        self.cluster_slider = ttk.Scale(
            slider_frame,
            from_=2,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.clusters_var,
            command=self.update_cluster_label
        )
        self.cluster_slider.pack(fill=tk.X, pady=5)
        
        self.cluster_value_label = tk.Label(slider_frame, text="4", 
                                         font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.cluster_value_label.pack()
        
        # Process button
        self.process_btn = ttk.Button(
            control_panel,
            text="Process Image",
            command=self.process_image,
            state=tk.DISABLED,
            style="Accent.TButton"
        )
        self.process_btn.pack(fill=tk.X, ipady=8, pady=10)
        
        # Status label
        self.status_label = tk.Label(control_panel, text="Ready", 
                                  font=("Arial", 10), fg="#555555", bg="#f0f0f0")
        self.status_label.pack(pady=10)
        
        # Help text
        help_text = "Upload an image and adjust the number of clusters to segment the image using Agglomerative Clustering."
        help_label = tk.Label(control_panel, text=help_text, wraplength=230, 
                           font=("Arial", 10), fg="#777777", bg="#f0f0f0")
        help_label.pack(pady=(50, 0))
        
        # Images display area (right side)
        self.display_frame = Frame(main_frame, bg="#f0f0f0")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create display containers
        self.original_frame = Frame(self.display_frame, bg="#ffffff", bd=1, relief=tk.SOLID)
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.result_frame = Frame(self.display_frame, bg="#ffffff", bd=1, relief=tk.SOLID)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Labels for the frames
        orig_title = tk.Label(self.original_frame, text="Original Image", 
                           font=("Arial", 14, "bold"), bg="#ffffff")
        orig_title.pack(pady=10)
        
        result_title = tk.Label(self.result_frame, text="Segmented Image", 
                             font=("Arial", 14, "bold"), bg="#ffffff")
        result_title.pack(pady=10)
        
        # Image containers
        self.original_canvas = tk.Canvas(self.original_frame, bg="white")
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.result_canvas = tk.Canvas(self.result_frame, bg="white")
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Configure style
        self.setup_styles()
    
    def setup_styles(self):
        # Configure ttk styles for modern look
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12))
        style.configure("TScale", sliderlength=20)
        
        # Create a custom style for accent buttons
        style.configure("Accent.TButton", 
                      background="#4a7abc", 
                      foreground="white", 
                      font=("Arial", 12, "bold"))
    
    def update_cluster_label(self, event=None):
        self.cluster_value_label.config(text=str(self.clusters_var.get()))
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return
            
        try:
            self.original_img = Image.open(file_path).convert('RGB')
            
            # Update file info
            file_name = os.path.basename(file_path)
            w, h = self.original_img.size
            self.file_info.config(text=f"{file_name}\n{w}x{h} pixels")
            
            # Enable process button
            self.process_btn.config(state=tk.NORMAL)
            
            # Display original image
            self.display_image(self.original_img, self.original_canvas)
            
            # Clear segmented image if any
            self.result_canvas.delete("all")
            self.processed_img = None
            
            self.status_label.config(text="Image loaded successfully")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
    
    def process_image(self):
        if self.original_img is None:
            return
            
        self.status_label.config(text="Processing...")
        self.root.update()
        
        try:
            n_clusters = self.clusters_var.get()
            segmented_img = agglomerative_clustering(self.original_img, n_clusters)
            self.processed_img = Image.fromarray(segmented_img)
            
            # Display segmented image
            self.display_image(self.processed_img, self.result_canvas)
            
            self.status_label.config(text=f"Segmentation complete with {n_clusters} clusters")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
    
    def display_image(self, img, canvas):
        canvas.delete("all")
        
        # Calculate the appropriate size for display
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 400
        
        img_width, img_height = img.size
        
        # Calculate the scaling factor to fit the image in the canvas
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        display_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)
        
        # Keep a reference to prevent garbage collection
        canvas.image = photo
        
        # Calculate position to center the image
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Display image on canvas
        canvas.create_image(x, y, image=photo, anchor=tk.NW)

def downsample_image(image, max_size=200):
    """Resize image to make clustering feasible"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def agglomerative_clustering(image, n_clusters=4):
    # Downsample the image to make computation feasible
    small_img = downsample_image(image)
    
    # Convert to array and reshape
    img_array = np.array(small_img)
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, c)
    
    # Normalize pixel values
    scaler = StandardScaler()
    pixels_normalized = scaler.fit_transform(pixels)
    
    # Use scikit-learn's AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'  # Ward's method tends to work well for images
    )
    
    # Fit clustering model
    labels = clustering.fit_predict(pixels_normalized)
    
    # Compute centroids manually
    centroids = np.zeros((n_clusters, c))
    for i in range(n_clusters):
        mask = labels == i
        centroids[i] = np.mean(pixels[mask], axis=0)
    
    # Map each pixel to its centroid color
    segmented = np.zeros_like(pixels)
    for i in range(n_clusters):
        mask = labels == i
        segmented[mask] = centroids[i]
    
    # Reshape back to image dimensions
    segmented_img = segmented.reshape(h, w, c).astype(np.uint8)
    
    # Resize back to original size
    if small_img.size != image.size:
        segmented_pil = Image.fromarray(segmented_img)
        segmented_pil = segmented_pil.resize(image.size, Image.NEAREST)
        segmented_img = np.array(segmented_pil)
    
    return segmented_img

def main():
    root = tk.Tk()
    app = AgglomerativeClusteringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
