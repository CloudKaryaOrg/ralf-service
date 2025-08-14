# ralf-frontend

This is the frontend interface for RALF framework to manage the LLM applications in enterprises. It is the acronym for: Recommend, Augmentation, Lustration, and Futureproofing.

## To build and run the image
```
docker build -t ralf-app .
docker run -p 8501:8501 --rm -t ralf-app
```

## To Do
1. Merge the Ralf class back to the CloudKaryaOrg/ralf repository
2. Split the application into 3 containers: frontend, eval-report, and training/fine-tuning
