# ralf-frontend

This is the frontend interface for RALF framework to manage the LLM applications in enterprises. It is the acronym for: Recommend, Augmentation, Lustration, and Futureproofing.

## To build and run the image
Option 1: Build and deploy using docker compose. The image is currently built with the intension to push to docker repository. After the build and deployment, a manual push is required.
**This should be modified to push to the private artifacts repository.
```
docker compose up
To shutdown
docker compose down
To push to image repository
docker push peeya/ralf-app
```
Option 2: Build to deploy on the current machine.
```
docker build -t ralf-app .
docker run -p 8501:8501 --rm -t ralf-app
```

## To Do
1. Merge the Ralf class back to the CloudKaryaOrg/ralf repository
2. Split the application into 3 containers: frontend, eval-report, and training/fine-tuning
