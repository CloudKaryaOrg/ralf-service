# Build for CloudRun deployment

To build the image for GCP CloudRun service, install the gcloud command: `brew install --cask google-cloud-sdk`

Update the `cloudbuild.yaml` file to the following:
```
    steps:
    - name: 'gcr.io/cloud-builders/docker'
      args: ['build', '-t', 'LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY/my-image-name:tag', '.']
    - name: 'gcr.io/cloud-builders/docker'
      args: ['push', 'LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY/my-image-name:tag']
```
Replace the following with your specific details:
* `LOCATION` with the region where the repository is located and where to deploy CloudRun.
* `PROJECT_ID` with your Google Cloud project ID.
* `REPOSITORY` with the name of your Artifact Registry repository.
* `my-image-name` with your image name.
* `tag` with a version tag (e.g., v1.0, latest).

Login to GCP: `gcloud init`\
or `gcloud auth login`

Build the image: `gcloud builds submit --config cloudbuild.yaml .`

Deploy the image:
```
gcloud run deploy SERVICE_NAME \
          --image LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/your-image-name:tag \
          --region SERVICE_REGION \
          --platform managed \
          --allow-unauthenticated # Or --no-allow-unauthenticated for authenticated access
```

Example:\
`cloudbuild.yaml`
```
    steps:
    - name: 'gcr.io/cloud-builders/docker'
      args: ['build', '-t', 'gcr.io/focal-woods-252003/ralf-service', 'ralf-fe']
    - name: 'gcr.io/cloud-builders/docker'
      args: ['push', 'gcr.io/focal-woods-252003/ralf-service']
```

Login: `gcloud auth login`

Set the project: `gcloud config set project focal-woods-252003`

Build the image: `gcloud builds submit --tag gcr.io/focal-woods-252003/ralf-eval-cr .`

Deploy the image:
```
gcloud run deploy ralf-service \
          --image gcr.io/focal-woods-252003/ralf-service \
          --region us-central1 \
          --platform managed \
          --port 8501 \
          --allow-unauthenticated
```