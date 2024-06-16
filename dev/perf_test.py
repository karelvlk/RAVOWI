import numpy as np
import asyncio
import time
import argparse
from src.image_content_recogniser import ImageContentRecogniser

# Assuming image_content_recogniser.process_perf_test_detection is an async function you have defined elsewhere


def print_result_table(total_time, avg_response_time, rps):
    # Define the data to be displayed
    data = [
        ("Total time for requests", f"{total_time:.2f} seconds"),
        ("Average response time", f"{avg_response_time:.4f} seconds"),
        ("Requests per second", f"{rps:.2f} RPS"),
    ]

    # Calculate the maximum width for the first column
    max_label_len = max(len(label) for label, _ in data) + 1  # Plus one for extra space

    # Calculate the maximum width for the second column
    max_value_len = max(len(value) for _, value in data)

    # Calculate total width of the table considering padding and separators
    total_width = (
        max_label_len + max_value_len + 5
    )  # 5 = 1 (left border) + 2 (space around) + 2 (right border)

    # Print the top border
    print("+" + "-" * (total_width - 1) + "+")

    # Print the title
    title = "Performance test results"
    print(
        f"| {title.center(total_width - 3)} |"
    )  # -4 = 2 (left padding) + 2 (right padding)

    # Print the separator
    print("+" + "-" * (total_width - 1) + "+")

    # Print each row of data
    for label, value in data:
        print(f"| {label.ljust(max_label_len)}: {value.ljust(max_value_len)} |")

    # Print the bottom border
    print("+" + "-" * (total_width - 1) + "+")


async def send_request(semaphore, image_content_recogniser, batch):
    """Function to send a single request and return the response time."""
    async with semaphore:  # This will limit the number of concurrent send_request calls
        images_batch = np.random.randint(
            low=0, high=256, size=(batch, 640, 640, 3), dtype=np.uint8
        )
        start_time = time.time()
        response = await image_content_recogniser.process_perf_test_detection(
            images_batch
        )
        end_time = time.time()
        return end_time - start_time


async def perf(NUM_REQUESTS, CONCURRENT_REQUESTS, BATCH_SIZE):
    print(
        f"Sending {NUM_REQUESTS} requests with {CONCURRENT_REQUESTS} concurrent requests in batch of {BATCH_SIZE}."
    )
    image_content_recogniser = ImageContentRecogniser()
    semaphore = asyncio.Semaphore(
        CONCURRENT_REQUESTS
    )  # Limit the number of concurrent requests
    start_time = time.time()
    tasks = [
        asyncio.create_task(
            send_request(semaphore, image_content_recogniser, BATCH_SIZE)
        )
        for _ in range(NUM_REQUESTS)
    ]
    response_times = await asyncio.gather(*tasks)
    end_time = time.time()

    total_time = end_time - start_time
    avg_response_time = sum(response_times) / len(response_times)
    rps = NUM_REQUESTS / total_time

    print(
        f"[ARGS] num_requests: {NUM_REQUESTS}, concurrent_requests: {CONCURRENT_REQUESTS}, batch_size: {BATCH_SIZE}"
    )
    print_result_table(total_time, avg_response_time, rps)


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run performance test for image content recognition."
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=1,
        help="Number of total requests to send.",
    )
    parser.add_argument(
        "--concurrent_requests",
        type=int,
        default=1,
        help="Number of concurrent requests.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images in batch.",
    )

    args = parser.parse_args()

    asyncio.run(perf(args.num_requests, args.concurrent_requests, args.batch_size))
