# Opencv_task
Face Recognition with OpenCV

This project implements face recognition using Python and OpenCV, specifically the LBPH (Local Binary Patterns Histograms) algorithm. The program detects faces from labeled training images, trains a model, and displays recognition results in a clean, labeled image grid.

⸻

Features:
	•	Face detection using Haar Cascades.
	•	Face recognition using the LBPH algorithm.
	•	Grid display of recognized faces.
	•	Each face is labeled with the person’s name, styled with a green box and centered black text.
	•	Works offline, easy to extend with more images.

⸻

Technologies Used:
	•	Python 3.8 or higher
	•	OpenCV (via opencv-contrib-python)
	•	NumPy
	•	Visual Studio Code (recommended IDE)
	•	Anaconda (optional, for managing the Python environment)

How It Works:
	•	The script loads all images inside the actor folders.
	•	It detects faces and assigns each person a label.
	•	The LBPH algorithm is trained using those faces.
	•	Then it loops over all test images, predicts the name, and renders a green label above each face.
	•	All results are displayed in a single grid window for easy viewing.

⸻

Usage Notes:
	•	Only .png images are supported by default.
	•	All images must contain a clear, front-facing face.
	•	The system is designed to work with at least one image per actor folder.
	•	Labels will automatically be drawn above detected faces.

⸻

Optional Enhancements:
	•	Add .jpg support in the file loader.
	•	Save the final output grid as an image (e.g., grid_output.jpg).
	•	Add live webcam recognition using OpenCV’s VideoCapture.

⸻

License:

This project is intended for educational and personal learning use. You are free to adapt or extend it.
