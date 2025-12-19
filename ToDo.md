1. Data Engeneering and Signal Processing
   Goal:
   Understend the IMS/NASA dataset and establish a data processing pipeline.
   Task:
   Ingest raw NASA sata files.
   Implement FFT(Fast Fourier Transform) and STFT(Short-Time Fourier Transform) in Python.
   Generate Mel-Spectrograms.
   KPI:
   A script that accepts a .csv file as input and outputs a tensor ready for neaural network ingestion.

2. Data Science and ML Modeling
   Goal:
   Build the 'Brain' of the system.
   Task:
   Develop and Autoencoder atchitecture (Conv@D or LSTM) using TensorFlow.
   Train the model exclusively on "healthy" data (Learning Normalcy).
   Determine the reconstruction error threshold for anomaly detection.
   Export the model to ONNX format (crucial for portability).

3. Cloud Simulation & Infrastructure
   Goal:
   Construct the runtime environment.
   Tasks:
   Configure docker-compose with LocalStack (mimicking S3 + Lambda).
   Develop the AWS Lambda code (Python) to load the ONNX model and perform inference.
   Integration Testing: File upload to S3 -> Lambda Trigger -> Inference result in logs.

4. Integration & Visualization
   Goal:
   End-to-end integration and dashboard creation.
   Tasks:
   Edge Device Simulation: Create a script to simulate sending processed data packets to S3.
   Build a streamlined dashboard using Streamlit.
   Draft README documentation (covering both business context and technical implementation).
