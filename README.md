
# Bank Card Predictions - Machine Learning

This project focuses on predicting the type of card a customer may issue based on the characteristics of individuals using cards issued by a particular bank. It leverages machine learning techniques and a CI/CD structure to streamline the development and deployment process.


## Introduction

The goal of this project is to utilize machine learning to predict the card type a customer may issue based on their characteristics, including family information, marital status, and their card usage patterns. By leveraging predictive models, the bank can tailor its services to customer preferences and optimize marketing strategies.

## Features

-   Data preprocessing and transformation
-   Model training and prediction
-   Flask-based web application for interactive predictions
-   CI/CD structure for efficient development and deployment
-   Use of DVC for versioning and managing data pipelines

## Project Structure
- .github/workflows/.gitkeep
- src/
  - components/
    - data_ingestion
    - data_transformation
    - model_trainer
  - utils/
    - __init__.py
  - pipeline/
    - predict_pipeline
  - logger.py
  - exception.py
- config/
  - config.yaml
- Notebook/
  - trial.ipynb
- templates/
  - index.html
  - home.html
- dvc.yaml
- params.yaml
- requirements.txt
- setup.py
     

## Technologies Used

-   Python
-   Flask
-   DVC (Data Version Control)
-   Machine Learning Libraries (Scikit-Learn, Pandas, NumPy)
-   HTML/CSS (for web application)
-   CI/CD Tools (GitHub Actions, Jenkins, etc.)

## Getting Started

### Prerequisites

-   Python 3.6+
-   Pip package manager

### Installation

1.  Clone the repository:
    
    bashCopy code
    
    `git clone https://github.com/yourusername/bank-card-predictions.git` 
    
2.  Navigate to the project directory:
    
    bashCopy code
    
    `cd bank-card-predictions` 
    
3.  Create a virtual environment (recommended):
    
    bashCopy code
    
    `python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate` 
    
4.  Install project dependencies:
    
    bashCopy code
    
    `pip install -r requirements.txt` 
    

## Usage

1.  Data Ingestion and Transformation:
    
    Run the data ingestion and transformation stages using DVC:
    
    bashCopy code
    
    `dvc run data_ingestion
    dvc run data_transformation` 
    
2.  Model Training:
    
    Train the machine learning model using DVC:
    
    bashCopy code
    
    `dvc run model_training` 
    
3.  Start the Flask Web Application:
    
    Run the Flask web application:
    
    bashCopy code
    
    `python app.py` 
    
    Access the application in your web browser at `http://localhost:5000`.
    

## CI/CD

This project uses a CI/CD pipeline to automate build, test, and deployment processes. [GitHub Actions](https://docs.github.com/en/actions) is used as an example, but you can integrate with other CI/CD platforms as well.

## Web Application

The Flask-based web application allows users to interactively predict the type of card a customer may issue based on input data. Access the application by starting the Flask server and navigating to the provided URL.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests to contribute to the project's development.

## License

This project is licensed under the MIT License. See the [LICENSE](https://chat.openai.com/LICENSE) file for details.
