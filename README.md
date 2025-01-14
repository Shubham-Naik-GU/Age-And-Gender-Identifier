# Age and Gender Identifier

## Project Description
The **Age and Gender Identifier** project is a machine learning application that predicts a person's age and gender from images. The model is developed using **Convolutional Neural Networks (CNN)**, **TensorFlow**, and **Keras**, and was trained in **Jupyter Notebook** and **Google Colab**. The trained model is integrated into a user-friendly interface built using **Streamlit** in **PyCharm**, allowing users to upload images in real-time and get predictions.

The system uses two models:
1. The first model, trained on the initial dataset, predicts age and gender from the images.
2. The second model, trained on corrected data, is used when the first model provides inaccurate predictions.

Data visualization tools like **Matplotlib** and **Seaborn** are utilized to assess the model's performance. Users can upload multiple images at once for batch processing, and the system will provide predictions from both models.

### Features:
- **Real-Time Image Uploads:** Upload multiple images and get predictions of age and gender.
- **Dual Model Prediction:** Two models are used to improve prediction accuracy, especially in cases of incorrect outputs.
- **Data Visualization:** **Matplotlib** and **Seaborn** are used to visualize and assess the model performance.
- **User Interface:** A simple UI created using **Streamlit** for easy interaction with the model.

## Installation Instructions

### 1. Dataset
- Download the **UTKFace** dataset, which contains images labeled with age, gender, and ethnicity.

### 2. Requirements
#### For training the model (in **Jupyter Notebook** or **Google Colab**):
- `tensorflow`
- `numpy`
- `pandas`
- `random`
- `matplotlib`
- `seaborn`

#### For the user interface in **PyCharm**:
- `streamlit`
- `tensorflow`
- `numpy`

### 3. Setup Instructions:
1. Download the **UTKFace** dataset.
2. Set up **Jupyter Notebook** or **Google Colab** for model training.
3. Install the necessary dependencies in the respective environments.
4. After training the models, proceed to set up the user interface in **PyCharm**.

## Usage

### 1. Training the Model
To train the models, follow these steps:
1. Open the `agengender (2).ipynb` file in **Jupyter Notebook** or **Google Colab**.
2. Run the cells in sequence to train the models.
3. After running the notebook, two trained models will be created.

**Note:** Make sure to adjust the file paths based on your system setup.

### 2. Running the User Interface
1. Open the `apk.py` file in **PyCharm**.
2. Run the following command in the terminal:
   ```bash
   streamlit run apk.py


