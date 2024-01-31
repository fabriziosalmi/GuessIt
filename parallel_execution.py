import concurrent.futures
from progress_monitor import log_info

def parallel_process(function, data, max_workers=5):
    """
    Execute a function in parallel over a set of data.
    :param function: The function to execute.
    :param data: Iterable of data to process.
    :param max_workers: Maximum number of worker threads.
    :return: List of results.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {executor.submit(function, item): item for item in data}
        for future in concurrent.futures.as_completed(future_to_data):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                log_info(f'Generated an exception: {exc}')
    return results

def sample_function(item):
    """
    A sample function that could be executed in parallel. Replace with actual function.
    :param item: Single data item to process.
    :return: Processed item.
    """
    # Process item
    return item

# Example Usage:
# data_to_process = [data1, data2, data3, ...]
# results = parallel_process(sample_function, data_to_process)
