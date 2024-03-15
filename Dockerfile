FROM python:3.10
# Set environment variables
ENV https_proxy=http://proxy.houston.hpecorp.net:8080
ENV HTTPS_PROXY=http://proxy.houston.hpecorp.net:8080
ENV HTTP_PROXY=http://proxy.houston.hpecorp.net:8080
ENV http_proxy=http://proxy.houston.hpecorp.net:8080
# Create app directory
WORKDIR /app
# Bundle app source
COPY chromadb /app
COPY main.py /app
# Install libs
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
# open ports
EXPOSE 7865
CMD [ "python", "main.py" ]
