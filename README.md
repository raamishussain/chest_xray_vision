# chest_xray_vision

This repository contains a FastAPI application which allows for a user to classify a
Chest X-ray image as having a potentially cancerous mass or nodule in the lung.

To run the application and classify images, follow the steps outlined below:

## Run FastAPI application locally

To run the application locally, first open a terminal and make sure Python 3.9 is installed.

Then pip install `pyenv` using the following command:

```
pip install pyenv
```

Once `pyenv` is install, create a virtual environment for this project:

```
pyenv virtualenv 3.9.13 chest_xray_vision
```

then activate the virtual environment:

```
pyenv activate chest_xray_vision
```

Once your virtual environment is ready, pip install the requirements of the project by running the following command in the terminal:

```
pip install -r requirements.txt
```

Now you can run the FastAPI application using the following command:

```
uvicorn chest_xray_vision.app:app --host "0.0.0.0" --port "8000"
```

Now open up a web browser and navigate to `http://localhost:8000/docs`. You should see the following page:

[swagger_ui](swagger_ui.png)

Click on the green `POST` button to expand the API. Then click on `Try it out` on the right side. Now you can choose an image file from the provided images and upload it to the application. Then click `Execute` to run the model.

The model prediction should appear below under `Response body`.