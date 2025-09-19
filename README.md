# Real-Time Emotion Detection Web App

A web application that uses deep learning to detect and classify human emotions from facial expressions in real time. Users can either upload an image or use their webcam. Includes visualization tools such as training history plots and a confusion matrix for evaluation.

---

## Features

- Real-time emotion detection via webcam feed  
- Emotion detection from uploaded images  
- CNN-based emotion classifier trained on facial expression data  
- Visualizations: training loss & accuracy history, confusion matrix  
- Web frontend built with Flask with HTML/JS integration  

---

## Emotion Labels

| Label Index | Emotion    |
|-------------|-------------|
| 0           | Angry       |
| 1           | Disgusted   |
| 2           | Fearful     |
| 3           | Happy       |
| 4           | Neutral     |
| 5           | Sad         |
| 6           | Surprised   |

---

## Technologies Used

- Python (>= 3.7)  
- Flask  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib & Seaborn  
- HTML, JavaScript  

---

## Directory Structure

```

emotion-detection-app/
├── app.py                     # Main Flask application
├── emotion-detect.keras       # Pre-trained CNN model
├── history.json               # Training history (accuracy & loss per epoch)
├── confusion\_matrix.json      # Confusion matrix data
├── templates/
│   └── trial.html             # Front-end HTML + JS for uploads and webcam
├── uploads/                   # Folder to store uploaded images
└── requirements.txt           # Python dependencies

```

---

## Setup / Installation

1. **Clone the repository**
```

git clone [https://github.com/Monish21072004/Emotion-detection-web-](https://github.com/Monish21072004/Emotion-detection-web-)
cd Emotion-detection-web-

````

2. **Create a virtual environment (optional but recommended)**
```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
````

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

   By default, the Flask server will start (commonly on `http://127.0.0.1:5000/`).

---

## Usage

* Use your webcam: The app will access your webcam and display live emotion detection.
* Upload an image: Use the upload interface to detect emotion from static images.
* View visualizations:

  * *Training history* (accuracy & loss over epochs)
  * *Confusion matrix* to see how well the model distinguishes each emotion

---

## Model & Training

* The pretrained model is stored in `emotion-detect.keras`.
* Training history is saved in `history.json`.
* Confusion matrix data in `confusion_matrix.json`.

If you want to retrain or improve the model, you would need the original dataset and modify the training scripts accordingly.

---

## Requirements

* Python 3.7 or newer
* All Python dependencies listed in `requirements.txt`

---

## Contributing

If you’d like to contribute:

1. Fork the repository
2. Create a branch for your feature or bugfix
3. Ensure your changes are tested
4. Submit a pull request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, feedback, or issues, contact: *\[Monish V / monishv217@gmail.com]*



