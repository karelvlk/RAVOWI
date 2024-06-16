#!/bin/bash

# Replace with your directory path
image_dir="/Users/karelvlk/Downloads/icr-uniques/"

# Initialize variables for time estimation
start_time=$(date +%s)
total_files=$(ls -1q "$image_dir"/*.jpg | wc -l)
count=0

# Loop through all jpg files in the directory
for image_path in "$image_dir"/*.jpg; do
  # Increment counter
  ((count++))

  # Check if file exists
  if [ ! -f "$image_path" ]; then
    echo "File $image_path does not exist."
    continue
  fi

  # Send POST request using curl
  response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:4000/validator/detect-json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@$image_path;type=image/jpeg")

  # Log success or failure
  if [ "$response" -eq 200 ]; then
    echo "Success for image $count/$total_files: $image_path"
  else
    echo "Failed for image $count/$total_files: $image_path, HTTP status code: $response"
  fi

  # Estimate time till completion
  elapsed_time=$(($(date +%s) - start_time))
  remaining_files=$((total_files - count))
  estimated_time=$((elapsed_time / count * remaining_files))

  # Convert estimated time to minutes and seconds
  estimated_min=$((estimated_time / 60))
  estimated_sec=$((estimated_time % 60))

  echo "Estimated time till completion: $estimated_min minutes and $estimated_sec seconds"
done

echo "All done! 🎉"
