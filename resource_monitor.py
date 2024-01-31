import psutil
from progress_monitor import log_info

def get_cpu_usage():
    """
    Get current CPU usage.
    :return: CPU usage percentage.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    log_info(f"Current CPU usage: {cpu_usage}%")
    return cpu_usage

def get_memory_usage():
    """
    Get current memory usage.
    :return: Memory usage in percentage.
    """
    memory_usage = psutil.virtual_memory().percent
    log_info(f"Current Memory usage: {memory_usage}%")
    return memory_usage

def monitor_system_resources():
    """
    Monitor and log system resources (CPU and memory usage).
    """
    cpu_usage = get_cpu_usage()
    memory_usage = get_memory_usage()
    log_info(f"System Resources - CPU: {cpu_usage}%, Memory: {memory_usage}%")

# Example Usage:
# monitor_system_resources()
