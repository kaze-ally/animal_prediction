# 🐾 Animal Prediction Using Deep Learning

This project uses a deep learning model to classify or predict animals based on provided data inputs. Built using PyTorch, it includes a clean application interface for making predictions easily.

## 📂 Project Structure

```
animal_model/
├── animal_prediction.ipynb   # Model development and experimentation notebook
├── app.py                    # Main application script
├── model_helper.py           # Helper functions for loading and predicting
├── requirements.txt          # Project dependencies
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/animal_model.git
cd animal_model
```

### 2. Set Up Environment

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

If using Streamlit:

```bash
streamlit run app.py
```


## 🧠 Model Details

- Framework: **PyTorch**
- Trained model saved as it is not present in the repository .It can be create by running animal_prediction.ipynb : `saved_model.pth`
- Functions in `model_helper.py` manage model loading and prediction

## 📓 Jupyter Notebook

The `animal_prediction.ipynb` notebook contains data preprocessing, model training, evaluation, and insights from experimentation.

## 🛠️ Technologies Used

- Python
- PyTorch
- Streamlit
- Jupyter Notebook

## 📦 Requirements

To install project dependencies:

```bash
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the MIT License.

## 🙌 Acknowledgements

- PyTorch team for their deep learning tools
- Open source contributors and libraries
- dataset link:
https://www.kaggle.com/datasets/alessiocorrado99/animals10

