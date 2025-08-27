# ralf-frontend

This is the frontend interface for RALF framework to manage the LLM applications in enterprises. It is the acronym for: Recommend, Augmentation, Lustration, and Futureproofing.

## To build and run the image
Option 1: Build and deploy using docker compose. The image is currently built with the intension to push to docker repository. After the build and deployment, a manual push is required.
**This should be modified to push to the private artifacts repository.
```
# To build the image using docker compose. This command builds a universal image even if it is executed by the macos
docker compose build --no-cache ralf-app

To run the container
docker compose up

To shutdown
docker compose down

To push to image repository
docker push peeya/ralf-app
```
Option 2: Build to deploy on the current machine.
```
docker build -t ralf-app ralf-fe

To build container that will run on linux and macos
docker buildx build --platform linux/amd64 -t ralf-app ralf-fe

docker run -p 8501:8501 --rm -t ralf-app
```

## To Do
1. Merge the Ralf class back to the CloudKaryaOrg/ralf repository
2. Split the application into 3 containers: frontend and eval-report

## Completed
1. Split training/finetuning functions into another class 