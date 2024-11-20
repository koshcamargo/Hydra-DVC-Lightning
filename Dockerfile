# Используем базовый образ Python
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    cmake \
    g++ \
    python3-dev \
    && apt-get clean

# Установка Python-библиотек
RUN pip install dvc[all] boto3 torch pytorch-lightning hydra-core numpy pybind11 build

# Установка рабочей директории
WORKDIR /app

# Копирование всех файлов проекта
COPY . /app

# Сборка C++ модуля
RUN cmake . && make && pip install .

# Установка Python-пакета
RUN python -m build && pip install dist/*.whl

# Инициализация DVC и загрузка данных
RUN dvc init \
    && dvc remote add -d myremote s3://mybucket/path \
    && dvc pull

# Запуск тестов
RUN python test_rowmean.py

# Запуск обучения
CMD ["python", "train.py"]
