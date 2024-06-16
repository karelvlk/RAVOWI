#!/bin/sh

IMAGENAME="meteval"
HOST_PORT=$2
CONTAINER_PORT=$3
FORCE_BUILD=$4

CONTAINERNAME="container-$IMAGENAME-$1"

# Function to build the image
build_image() {
    echo "Building image $IMAGENAME:latest..."
    docker build --rm -f docker/train/Dockerfile.cuda -t "$IMAGENAME:latest" .
}

# Check if --build argument is passed or image is not built
if [ "$FORCE_BUILD" = "--build" ] || [ "$(docker images -q $IMAGENAME:latest 2> /dev/null)" = "" ]; then
    build_image
fi

# Check if container is not running
if [ "$(docker ps -q -f name=$CONTAINERNAME)" = "" ]; then
  if [ "$(docker ps -aq -f status=exited -f name=$CONTAINERNAME)" != "" ]; then
    # cleanup
    echo "Removing exited container $CONTAINERNAME"
    docker rm $CONTAINERNAME
  fi
  # run your container
  echo "Running new container $CONTAINERNAME"
  docker run --rm -d -p $HOST_PORT:$CONTAINER_PORT -v $(pwd)/../../:/meteval --name "$CONTAINERNAME" -it --gpus all "$IMAGENAME:latest" 
else
  echo "Container $CONTAINERNAME is already running"
fi

# Exec into the running container
if [ "$(docker ps -q -f name=$CONTAINERNAME)" != "" ]; then
  echo "Entering container $CONTAINERNAME"
  docker exec -it "$CONTAINERNAME" sh
else
  echo "No running container to enter"
fi
