# Skin Insight thesis backend

FastAPI API for skin-image classification, XAI explanations, authentication, and history stored in MongoDB.

## Docker Compose

Run the API and MongoDB from this directory:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000` and its OpenAPI docs at `http://localhost:8000/docs`. Set `API_PORT=8001` to use another host port. Set `SECRET_KEY` to your own value when keeping user accounts between runs. MongoDB data is stored in the `mongo_data` volume.

The classification and XAI endpoints require `app/classification_models/resnet_model.h5`. Docker mounts `app/classification_models` read-only, so place that file on the host without rebuilding the image. If it is absent, the API still starts and those endpoints return HTTP 503. Other `*.weights.h5` files are retained locally but are not used by this version of the API.

To run the frontend, API, and MongoDB together, start Compose from `../skin-insight-thesis-frontend` instead. Run only one of the two Compose stacks at a time because both use the API port.
