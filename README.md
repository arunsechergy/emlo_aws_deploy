# Run a Gradio app as part of Docker Container
This is a two step process
Step 1: Build the docker image
```
docker build -t build_image:latest .
```

Step 2: Run the image as a container. I want to expose port 80
```
docker run -it -p 80:7860 build_image:latest
```
