import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk, Frame
import os
import time
import heapq  # For priority queue optimization
import cv2  # For better color space handling
from sklearn.preprocessing import StandardScaler  # For normalizing pixel values
from sklearn.cluster import AgglomerativeClustering  # For built-in clustering

class AgglomerativeClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation with Agglomerative Clustering")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        self.original_img = None
        self.processed_img = None
        self.use_builtin = False  # Default to custom implementation
        
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
            to=10,  # Reduced max clusters due to computational intensity
            orient=tk.HORIZONTAL,
            variable=self.clusters_var,
            command=self.update_cluster_label
        )
        self.cluster_slider.pack(fill=tk.X, pady=5)
        
        self.cluster_value_label = tk.Label(slider_frame, text="4", 
                                         font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.cluster_value_label.pack()
        
        # Method selection
        method_frame = Frame(control_panel, bg="#f0f0f0")
        method_frame.pack(fill=tk.X, pady=20)
        
        method_label = tk.Label(method_frame, text="Method:", 
                              font=("Arial", 12), bg="#f0f0f0")
        method_label.pack(anchor=tk.W)
        
        self.method_var = tk.StringVar(value="From Scratch")
        method_options = ["From Scratch", "Built-in"]
        self.method_menu = ttk.OptionMenu(
            method_frame, 
            self.method_var, 
            self.method_var.get(), 
            *method_options, 
            command=self.toggle_method
        )
        self.method_menu.pack(fill=tk.X, pady=5)
        
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
        
        # Progress bar
        self.progress = ttk.Progressbar(control_panel, orient=tk.HORIZONTAL, 
                                      length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # Time label
        self.time_label = tk.Label(control_panel, text="Time: 0.00 seconds", 
                                  font=("Arial", 10), fg="#555555", bg="#f0f0f0")
        self.time_label.pack(pady=10)
        
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
    
    def toggle_method(self, selected_value=None):
        """Toggle between custom and built-in implementation
        
        Parameters:
        selected_value: The value selected in the OptionMenu (passed automatically)
        """
        self.use_builtin = (self.method_var.get() == "Built-in")
        method_name = "Built-in (scikit-learn)" if self.use_builtin else "From Scratch"
        self.status_label.config(text=f"Using {method_name} implementation")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return
            
        try:
            self.original_img = Image.open(file_path).convert('RGB')
            
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
        self.progress["value"] = 0
        self.root.update()
        
        try:
            n_clusters = self.clusters_var.get()
            
            # Disable UI elements during processing
            self.process_btn.config(state=tk.DISABLED)
            self.upload_btn.config(state=tk.DISABLED)
            
            # Process in chunks to update UI
            def update_progress(value):
                self.progress["value"] = value
                self.root.update()
            
            start_time = time.time()
            
            if self.use_builtin:
                segmented_img = agglomerative_clustering_builtin(
                    self.original_img, 
                    n_clusters, 
                    progress_callback=update_progress
                )
            else:
                segmented_img = agglomerative_clustering_scratch(
                    self.original_img, 
                    n_clusters, 
                    progress_callback=update_progress,
                    enhanced_output=True
                )
            
            elapsed_time = time.time() - start_time
            self.time_label.config(text=f"Time: {elapsed_time:.2f} seconds")
            
            self.processed_img = Image.fromarray(segmented_img)
            
            # Display segmented image
            self.display_image(self.processed_img, self.result_canvas)
            
            self.status_label.config(text=f"Segmentation complete with {n_clusters} clusters")
            
            # Re-enable UI elements
            self.process_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.process_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)
    
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

def downsample_image(image, max_size=100):  # Increased size for better quality
    """Resize image to make clustering feasible"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def calculate_initial_distances(data):
    """Calculate initial pairwise distances between clusters more efficiently"""
    n = len(data)
    # Use a heap to store distances efficiently
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            dist = euclidean_distance(data[i], data[j])
            # Store as (distance, i, j) for heapq
            heapq.heappush(distances, (dist, i, j))
            
    return distances

def ward_linkage_distance_optimized(c1_size, c2_size, c1_center, c2_center):
    """Optimized Ward's linkage criterion using pre-calculated centroids"""
    if c1_size == 0 or c2_size == 0:
        return float('inf')
    
    # Ward's criterion formula
    return ((c1_size * c2_size) / (c1_size + c2_size)) * euclidean_distance(c1_center, c2_center)**2

def agglomerative_clustering_scratch(image, n_clusters=4, progress_callback=None, enhanced_output=True):
    """Optimized agglomerative clustering with improved visual quality"""
    start_time = time.time()
    
    # Store original image for later use
    original_img_array = np.array(image)
    
    # Downsample the image
    small_img = downsample_image(image, max_size=150)  # Increase size for better details
    
    # Convert to array and reshape
    img_array = np.array(small_img)
    h, w, c = img_array.shape
    
    # Convert to LAB color space for perceptually meaningful clustering
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    pixels = img_lab.reshape(-1, c)
    
    # Store RGB pixels for later color assignment
    rgb_pixels = img_array.reshape(-1, c)
    
    # Normalize pixel values - very important for LAB space
    # L: 0-100, a: -127 to 127, b: -127 to 127
    # We'll scale them to have similar importance
    pixels_normalized = pixels.astype(float)
    # Weight channels differently: L channel usually less important than a,b for segmentation
    weights = np.array([1.0, 2.0, 2.0])  # Give more weight to color than luminance
    scale = np.array([100.0, 127.0, 127.0])  # LAB value ranges
    pixels_normalized = (pixels_normalized / scale) * weights
    
    if progress_callback:
        progress_callback(10)
    
    # Number of data points
    n = len(pixels_normalized)
    
    # For very large images, sample a subset of pixels for clustering
    max_samples = 2000  # Increased for better representation
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        sampled_pixels = pixels_normalized[indices]
        is_sampled = True
    else:
        sampled_pixels = pixels_normalized
        indices = np.arange(n)
        is_sampled = False
    
    # Initialize clusters
    num_points = len(sampled_pixels)
    cluster_sizes = np.ones(num_points, dtype=int)
    cluster_centroids = sampled_pixels.copy()
    active_clusters = set(range(num_points))
    cluster_mapping = [{i} for i in range(num_points)]
    
    # Pre-compute initial distances
    distance_heap = calculate_initial_distances(sampled_pixels)
    valid_pairs = set((i, j) for _, i, j in distance_heap if i < j)
    
    # Continue until we reach the desired number of clusters
    current_clusters = num_points
    while current_clusters > n_clusters:
        if progress_callback:
            progress_percent = 10 + 80 * (1 - (current_clusters - n_clusters) / (num_points - n_clusters))
            progress_callback(min(90, progress_percent))
        
        # Find the closest pair of clusters
        while True:
            if not distance_heap:
                break
            
            dist, i, j = heapq.heappop(distance_heap)
            
            if i not in active_clusters or j not in active_clusters:
                continue
                
            if (i, j) not in valid_pairs:
                continue
                
            # Valid merge found
            break
        
        # Remove the pair from valid pairs
        valid_pairs.remove((i, j))
        
        # Get sizes and centroids for merging
        size_i = cluster_sizes[i]
        size_j = cluster_sizes[j]
        centroid_i = cluster_centroids[i]
        centroid_j = cluster_centroids[j]
        
        # Calculate the new centroid (weighted average)
        new_size = size_i + size_j
        new_centroid = (size_i * centroid_i + size_j * centroid_j) / new_size
        
        # Update the first cluster
        cluster_sizes[i] = new_size
        cluster_centroids[i] = new_centroid
        cluster_mapping[i].update(cluster_mapping[j])
        
        # Mark the second cluster as inactive
        active_clusters.remove(j)
        
        # Calculate new distances
        for k in active_clusters:
            if k != i:
                dist = ward_linkage_distance_optimized(
                    cluster_sizes[i], cluster_sizes[k],
                    cluster_centroids[i], cluster_centroids[k]
                )
                heapq.heappush(distance_heap, (dist, min(i, k), max(i, k)))
                valid_pairs.add((min(i, k), max(i, k)))
        
        current_clusters -= 1
    
    if progress_callback:
        progress_callback(90)
    
    # Create final clusters
    final_clusters = [cluster_mapping[i] for i in active_clusters]
    final_centroids = [cluster_centroids[i] for i in active_clusters]
    
    # Create a mapping from data points to cluster labels
    cluster_labels = np.zeros(len(sampled_pixels), dtype=int)
    for i, cluster in enumerate(final_clusters):
        for idx in cluster:
            cluster_labels[idx] = i
    
    # Map results back to original image size
    if is_sampled:
        # Assign each pixel to the nearest centroid
        full_labels = np.zeros(n, dtype=int)
        
        batch_size = 20000
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            batch = pixels_normalized[i:batch_end]
            
            # Calculate distance to each centroid
            distances = np.zeros((batch.shape[0], n_clusters))
            for j in range(n_clusters):
                diff = batch - final_centroids[j]
                distances[:, j] = np.sum(diff * diff, axis=1)
            
            # Assign to closest centroid
            full_labels[i:batch_end] = np.argmin(distances, axis=1)
    else:
        # For non-sampled data, use existing labels
        full_labels = cluster_labels
    
    # Create segmented image with realistic colors
    segmented = np.zeros_like(rgb_pixels)
    
    # For each cluster, compute the mean color from the original RGB values
    for i in range(n_clusters):
        mask = full_labels == i
        if np.any(mask):
            # Use the mean of original RGB values for this segment
            mean_color = np.mean(rgb_pixels[mask], axis=0)
            segmented[mask] = mean_color.astype(np.uint8)
    
    # Reshape back to image dimensions
    segmented_img = segmented.reshape(h, w, c)
    
    # Resize back to original size with better interpolation
    if small_img.size != image.size:
        segmented_pil = Image.fromarray(segmented_img)
        segmented_pil = segmented_pil.resize(image.size, Image.BICUBIC)
        segmented_img = np.array(segmented_pil)
    
    # Apply more gentle post-processing for better visuals
    # First, apply bilateral filter to smooth while preserving edges
    segmented_img = cv2.bilateralFilter(segmented_img, 9, 75, 75)
    
    # Then apply a more subtle edge-preserving filter
    segmented_img = cv2.edgePreservingFilter(segmented_img, flags=1, sigma_s=45, sigma_r=0.3)
    
    if progress_callback:
        progress_callback(100)
    
    elapsed_time = time.time() - start_time
    print(f"Clustering completed in {elapsed_time:.2f} seconds")
    
    return segmented_img

def agglomerative_clustering_builtin(image, n_clusters=4, progress_callback=None):
    """Use scikit-learn's built-in agglomerative clustering with improved visualization"""
    start_time = time.time()
    
    if progress_callback:
        progress_callback(10)
        
    # Downsample image to make computation feasible
    small_img = downsample_image(image, max_size=200)  # Larger size for built-in method
    
    # Convert to array and reshape
    img_array = np.array(small_img)
    h, w, c = img_array.shape
    rgb_pixels = img_array.reshape(-1, c)
    
    # Convert to LAB color space for better results
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    pixels_lab = img_lab.reshape(-1, c)
    
    # Normalize pixel values - with proper weighting for LAB
    weights = np.array([1.0, 2.0, 2.0])  # Give more weight to color than luminance
    scale = np.array([100.0, 127.0, 127.0])  # LAB value ranges
    pixels_weighted = (pixels_lab.astype(float) / scale) * weights
    
    if progress_callback:
        progress_callback(30)
    
    # Use scikit-learn's AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    
    # Fit model
    cluster_labels = model.fit_predict(pixels_weighted)
    
    if progress_callback:
        progress_callback(70)
    
    # Create segmented image with realistic colors
    segmented = np.zeros_like(rgb_pixels)
    
    # For each cluster, compute the mean color from original RGB values
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            mean_color = np.mean(rgb_pixels[mask], axis=0)
            segmented[mask] = mean_color.astype(np.uint8)
    
    # Reshape back to image dimensions
    segmented_img = segmented.reshape(h, w, c)
    
    # Resize back to original size with better interpolation
    if small_img.size != image.size:
        segmented_pil = Image.fromarray(segmented_img)
        segmented_pil = segmented_pil.resize(image.size, Image.BICUBIC)
        segmented_img = np.array(segmented_pil)
    
    # Post-processing for enhanced output - more gentle settings
    segmented_img = cv2.bilateralFilter(segmented_img, 9, 75, 75)
    segmented_img = cv2.edgePreservingFilter(segmented_img, flags=1, sigma_s=45, sigma_r=0.3)
    
    if progress_callback:
        progress_callback(100)
    
    elapsed_time = time.time() - start_time
    print(f"Built-in clustering completed in {elapsed_time:.2f} seconds")
    
    return segmented_img

def main():
    root = tk.Tk()
    app = AgglomerativeClusteringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
