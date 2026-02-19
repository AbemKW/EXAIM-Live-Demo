import os
import time
import logging
import lambda_cloud_client
from lambda_cloud_client.rest import ApiException
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User data script for cloud-init
# This script will be executed on the GPU instance upon launch
USER_DATA_SCRIPT = """#cloud-config
runcmd:
  - apt-get update
  - apt-get install -y curl python3-pip
  - pip install lms lmctl lambda-cloud-client requests

  # Install and start LM Studio
  - curl -L https://install.lmstudio.ai | bash
  - lms daemon up &
  - sleep 10 # Give LMS daemon time to start
  - lms get lmstudio-community/medgemma-27b-text-it-GGUF:q4km # Explicitly download the model
  - lms load lmstudio-community/medgemma-27b-text-it-GGUF:q4km --context-length 16384 # Load the model with 16k context length
  - lms server start --model lmstudio-community/medgemma-27b-text-it-GGUF:q4km --port 1234 &

  # Background Python script to monitor GPU idle time and self-terminate
  - |
    #!/usr/bin/env python3
    import time
    import subprocess
    import os
    import requests

    IDLE_TIMEOUT_SECONDS = 3 * 3600 # 3 hours
    POLL_INTERVAL_SECONDS = 60 # Check every minute
    LAMBDA_API_KEY = os.environ.get("LAMBDA_API_KEY")
    INSTANCE_ID = os.environ.get("LAMBDA_INSTANCE_ID") # Set by Lambda Labs during provisioning

    if not LAMBDA_API_KEY or not INSTANCE_ID:
        print("LAMBDA_API_KEY or LAMBDA_INSTANCE_ID not set. Skipping self-termination script.")
        exit(1)

    idle_start_time = None

    while True:
        try:
            # Check GPU utilization using nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            gpu_utilization = [int(x.strip()) for x in result.stdout.strip().split('\\n')]

            is_idle = all(util == 0 for util in gpu_utilization)

            if is_idle:
                if idle_start_time is None:
                    idle_start_time = time.time()
                    print(f"GPU is idle. Starting idle timer: {IDLE_TIMEOUT_SECONDS / 3600} hours.")
                elif (time.time() - idle_start_time) >= IDLE_TIMEOUT_SECONDS:
                    print(f"GPU has been idle for {IDLE_TIMEOUT_SECONDS / 3600} hours. Terminating instance...")
                    headers = {"Authorization": f"Bearer {LAMBDA_API_KEY}"}
                    response = requests.delete(f"https://cloud.lambdalabs.com/api/v1/instance/{INSTANCE_ID}", headers=headers)
                    response.raise_for_status()
                    print("Instance terminated successfully.")
                    break # Exit loop after termination
            else:
                if idle_start_time is not None:
                    print("GPU activity detected. Resetting idle timer.")
                idle_start_time = None

        except Exception as e:
            print(f"Error monitoring GPU: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)
"""

def provision_gpu(
    instance_type_name: str = "gpu_1x_a10",
    region_name: str = "us-west-1",
) -> Optional[str]:
    """
    Provisions a GPU instance using Lambda Labs, or returns the IP of an existing active instance.
    The instance will set up LM Studio and load the medgemma model.
    It will also include a self-termination script if idle for 3 hours.

    Returns:
        The IP address of the active GPU instance, or None if provisioning fails.
    """
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        logger.error("LAMBDA_API_KEY environment variable not set.")
        return None

    # Configure API key authorization: Bearer
    configuration = lambda_cloud_client.Configuration(
        host = "https://cloud.lambdalabs.com/api/v1" # Default host
    )
    configuration.access_token = api_key

    # Create an API client instance
    api_client = lambda_cloud_client.ApiClient(configuration)
    instance_api = lambda_cloud_client.InstanceApi(api_client)

    logger.info("Checking for existing active GPU instances...")
    try:
        # list_instances returns a ListInstancesResponse object which has an 'data' attribute
        # which is a list of Instance objects.
        list_response = instance_api.list_instances()
        instances = list_response.data
        for instance in instances:
            # Check if instance.name and instance.instance_type exist before accessing
            if instance.status == "active" and \
               instance.instance_type and \
               instance.instance_type.name == instance_type_name and \
               instance.ip_address:
                logger.info(f"Found existing active instance: {instance.id} at IP: {instance.ip_address}")
                return instance.ip_address
    except ApiException as e:
        logger.error(f"Exception when calling InstanceApi->list_instances: {e}")
        return None

    logger.info(f"No active instance found. Launching a new '{instance_type_name}' instance...")
    try:
        # Launch a new instance
        # The launch_instance method expects a LaunchInstanceRequest object
        launch_request = lambda_cloud_client.LaunchInstanceRequest(
            instance_type_name=instance_type_name,
            region_name=region_name,
            ssh_key_names=["ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCVhhrV8CN2R9/9PxAG+ynPws6wAWeE4Yr1SwwN9LfxpH4Y39a33QkOvLNx87za//xXV7c9/tVJXr/keJRKMVW2mkMPTL2sbMojnF5mu39SbOMCtgSKlmKTsi6jaK+zSFBVj91TnCZdE/EtfpBrS3E5PIGdY2NyLliTfMOX01RgHzEBfoGJMryeR0Nv9uxdHgPjlgqXfxJ/ID+odFhdIz0NnQbQxEe9GJomFjOUvBf0VsBIhLGajiSqlt/OLSyYkacDH+8S9G3rxQUxmI5Suzq5M/Evw50skX01BZKhGRs+SyZuf4RiYg9smSlPPb4hvQkE+FFSKO6jU83h7E6niy3P EXAIM-Live"], # IMPORTANT: Replace with your actual SSH key name
            file_system_names=[],
            name="exaim-gpu-instance",
            user_data=USER_DATA_SCRIPT,
        )
        new_instance_response = instance_api.launch_instance(launch_instance_request=launch_request)
        new_instance_id = new_instance_response.data[0].id # Assumes first instance in the response
        logger.info(f"Instance '{new_instance_id}' launched successfully.")

        # Wait for the instance to become active and get its IP
        max_attempts = 60
        for attempt in range(max_attempts):
            instance_response = instance_api.get_instance(new_instance_id)
            instance = instance_response.data # get_instance returns a GetInstanceResponse object
            if instance.status == "active" and instance.ip_address:
                logger.info(f"Instance '{instance.id}' is active with IP: {instance.ip_address}")
                return instance.ip_address
            logger.info(f"Waiting for instance '{instance.id}' to become active... (Attempt {attempt + 1}/{max_attempts})")
            time.sleep(10) # Wait 10 seconds before polling again

        logger.error(f"Instance '{new_instance_id}' did not become active within the expected time.")
        return None

    except ApiException as e:
        logger.error(f"Exception when calling InstanceApi->launch_instance or get_instance: {e}")
        return None
    except Exception as e:
        logger.error(f"Error launching or monitoring instance: {e}")
        return None

if __name__ == "__main__":
    # This block is for testing purposes.
    # In a real scenario, LAMBDA_API_KEY would be set in the environment.
    # os.environ["LAMBDA_API_KEY"] = "YOUR_LAMBDA_API_KEY"
    # os.environ["LAMBDA_INSTANCE_ID"] = "YOUR_INSTANCE_ID" # This would be set by Lambda during provisioning

    print("Attempting to provision GPU...")
    gpu_ip = provision_gpu()
    if gpu_ip:
        print(f"GPU instance IP: {gpu_ip}")
    else:
        print("Failed to provision GPU instance.")
