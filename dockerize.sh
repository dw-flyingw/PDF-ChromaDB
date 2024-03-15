docker build -t dave-pdf-chroma .
docker run --rm -it -p 7865:7865 dave-pdf-chroma
docker ps -a | grep dave
#docker stop dave-pdf-chroma
docker images -a | grep -i dave
#docker rmi dave-pdf-chroma