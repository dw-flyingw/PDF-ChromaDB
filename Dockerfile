FROM ubuntu:latest

RUN apt update && apt install -y openssh-server
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN useradd -m -s /bin/bash dave
RUN echo "dave:dave" | chpasswd

EXPOSE 22

ENTRYPOINT service ssh start && bash

