# Demo KAI (Kodak AI microservice)

# Introduction 
DemoKAI is a microservice that demonstrates the use of KAI (Kodak AI) in a KMMP environment. The service is built using WSL-Ubuntu to produce a Linux binary executable file that accepts an image file as input and outputs an image file overlayed with KAI results. Through integrating with Gradio, the demo allows users to upload and download image via http://localhost:7860/.


# Getting Started
1. Clone the repository: https://dev.azure.com/KodakMoments/KMPlatform/_git/DemoKAI
2. Switch out to the branch `<face_detection>`.
3. Copy `FacialImaging` folder to `<DemoKAI\src\KAI\FacialImaging>`.
4. Copy ML Config file `MLConfig_FD.json` to `<DemoKAI\src\KAI\FacialImaging\MLConfig_FD.json>`.
5. Make sure you have Docker installed on your WSL.

# Build and Test
1. Navigate to the project directory:
```
cd DemoKAI\src
```
2. Build the docker image:
```
build.sh
```
3.	Run the docker image:
```
docker compose up
``` 
Note: Some system use "docker-compose" and require the compose.yml file to be called "docker-compose.yml"

4. Using a web browser, navigate to http://localhost:7860/ to access the Gradio interface.
5. Upload an image file.
6. Press "Submit" button to see the KAI-processed image.
7. Press "Download" button to download the processed image.