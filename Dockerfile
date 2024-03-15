FROM python:3.10
# Create app directory
WORKDIR /app
# Bundle app source
COPY chromadb /app
COPY main.py /app
COPY requirements.txt /app
EXPOSE 7865
CMD [ "python", "main.py" ]
