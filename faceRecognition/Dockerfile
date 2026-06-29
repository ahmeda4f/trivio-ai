FROM python:3.10

WORKDIR /code

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN mkdir -p /root/.deepface/weights

COPY . /code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]