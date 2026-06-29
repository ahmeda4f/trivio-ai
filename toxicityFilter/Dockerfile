FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/app.py

RUN mkdir -p /code/cache && chmod -R 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]