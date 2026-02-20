import os
import httpx
import asyncio
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class LambdaLifecycleManager:
    """Manager for Lambda Cloud GPU instances lifecycle.
    
    Automates the discovery, provisioning, and readiness checking of GPU instances.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        self.base_url = "https://cloud.lambdalabs.com/api/v1"
        if not self.api_key:
            logger.warning("LAMBDA_API_KEY not found in environment or constructor.")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def get_existing_instance(self, instance_type: str = "gpu_1x_a10") -> Optional[Dict]:
        """Find an existing instance of the specified type (active or booting).
        
        Returns:
            Dict containing instance details if found, else None.
        """
        if not self.api_key:
            return None
            
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/instances", headers=self.headers)
                response.raise_for_status()
                instances = response.json().get("data", [])
                
                # Check for instances that are either active or booting
                for inst in instances:
                    if (inst.get("status") in ["active", "booting"] and 
                        inst.get("instance_type", {}).get("name") == instance_type):
                        return inst
        except Exception as e:
            logger.error(f"Error listing instances from Lambda API: {e}")
            
        return None

    async def provision_gpu(self, instance_type: str = "gpu_1x_a10", region: str = "us-west-1") -> Optional[str]:
        """Provisions a new GPU instance or returns the IP of an existing one.
        
        Args:
            instance_type: The Lambda Labs instance type name.
            region: The region name to launch in.
            
        Returns:
            The IP address of the active instance, or None if provisioning fails.
        """
        if not self.api_key:
            logger.error("Cannot provision GPU: LAMBDA_API_KEY is missing.")
            return None
            
        # 1. Detection: Find existing active or booting instance
        instance = await self.get_existing_instance(instance_type)
        instance_id = None
        
        if instance:
            instance_id = instance.get("id")
            if instance.get("status") == "active" and instance.get("ip_address"):
                ip = instance.get("ip_address")
                logger.info(f"Using existing active instance: {instance_id} at {ip}")
                return ip
            else:
                logger.info(f"Found existing instance {instance_id} with status '{instance.get('status')}'. Waiting for it to become active...")
        else:
            # 2. Launch Logic: Launch new instance if none exists
            logger.info(f"No active or booting instance found. Launching new '{instance_type}' in '{region}'...")
            
            # Get user data with embedded auto-termination
            user_data = self._get_user_data_script()
            
            # Get an SSH key (required for launch)
            ssh_key_name = await self._get_first_ssh_key()
            if not ssh_key_name:
                logger.error("No SSH keys found in Lambda Labs account. Launch will fail.")
                return None
                
            payload = {
                "region_name": region,
                "instance_type_name": instance_type,
                "ssh_key_names": [ssh_key_name],
                "file_system_names": [],
                "name": f"exaim-gpu-{instance_type.replace('_', '-')}",
                "user_data": user_data
            }
            
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.post(
                        f"{self.base_url}/instance-operations/launch",
                        json=payload,
                        headers=self.headers
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    
                    # Handle both "data" (list of objects) and "instance_ids" (list of strings) formats
                    launch_data = response_json.get("data", [])
                    if not launch_data and "instance_ids" in response_json:
                        launch_data = response_json["instance_ids"]
                        
                    if not launch_data:
                         logger.error(f"Launch response did not contain instance data: {response_json}")
                         return None
                         
                    # The first item in launch_data is our new instance
                    first_item = launch_data[0]
                    if isinstance(first_item, dict):
                        instance_id = first_item.get("id")
                    else:
                        instance_id = first_item # It's a string ID
                        
                    logger.info(f"Instance '{instance_id}' launch initiated.")
            except Exception as e:
                logger.error(f"Error during instance launch: {e}")
                return None

        # 3. Wait for IP and active status
        if not instance_id:
            return None
            
        logger.info(f"Waiting for instance '{instance_id}' to become active...")
        max_attempts = 120 # 20 minutes with 10s polling
        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(max_attempts):
                try:
                    resp = await client.get(f"{self.base_url}/instances/{instance_id}", headers=self.headers)
                    if resp.status_code == 200:
                        inst_data = resp.json().get("data", {})
                        if inst_data.get("status") == "active" and inst_data.get("ip_address"):
                            logger.info(f"Instance '{instance_id}' is now active at {inst_data['ip_address']}")
                            return inst_data["ip_address"]
                        elif inst_data.get("status") == "terminated":
                            logger.error(f"Instance '{instance_id}' was terminated unexpectedly.")
                            return None
                    else:
                        logger.warning(f"Failed to poll instance status (Attempt {attempt+1}): {resp.status_code}")
                except Exception as e:
                    logger.warning(f"Error polling instance status: {e}")
                
                await asyncio.sleep(10)
                        
        logger.error(f"Instance '{instance_id}' did not become active within the timeout period.")
        return None

    async def _get_first_ssh_key(self) -> Optional[str]:
        """Fetch available SSH keys and return the first one's name."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/ssh-keys", headers=self.headers)
                response.raise_for_status()
                keys = response.json().get("data", [])
                if keys:
                    return keys[0].get("name")
        except Exception as e:
            logger.error(f"Error fetching SSH keys: {e}")
        return None

    def _get_user_data_script(self) -> str:
        """Generates the cloud-init user_data script for instance provisioning."""
        return f"""#cloud-config
runcmd:
  - apt-get update
  - apt-get install -y curl python3-pip
  - pip install requests

  # Install and start LM Studio CLI (lms)
  - curl -fsSL https://lmstudio.ai/install.sh | bash
  - export PATH="$PATH:$HOME/.cache/lm-studio/bin" # Ensure lms is in path
  - lms daemon up &
  - sleep 10

  # Start LM Studio server and load model
  - lms server start --port 1234 &
  - sleep 5
  - lms get lmstudio-community/medgemma-27b-text-it-GGUF:q4km
  - lms load medgemma-27b-text-it-GGUF:q4km --context-length 16384

  # Background Python script for auto-termination after 3 hours of idle GPU
  - |
    cat <<'EOF' > /usr/local/bin/auto_terminate.py
    import time
    import subprocess
    import os
    import requests

    IDLE_TIMEOUT_SECONDS = 3 * 3600 # 3 hours
    POLL_INTERVAL_SECONDS = 60
    LAMBDA_API_KEY = "{self.api_key}"
    TERMINATE_URL = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"

    def get_my_instance_id():
        '''Finds our own instance ID by matching our public IP with the API list.'''
        try:
            # Try to get public IP
            ip = requests.get("https://api.ipify.org", timeout=10).text.strip()
            headers = {{"Authorization": f"Bearer {{LAMBDA_API_KEY}}"}}
            resp = requests.get("https://cloud.lambdalabs.com/api/v1/instances", headers=headers, timeout=10)
            if resp.status_code == 200:
                instances = resp.json().get("data", [])
                for inst in instances:
                    if inst.get("ip_address") == ip:
                        return inst.get("id")
        except Exception as e:
            print(f"Error identifying own instance ID: {{e}}")
        return None

    def check_gpu_idle():
        '''Returns True if GPU utilization is 0%.'''
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            # Check if all GPUs are idle
            utilizations = [int(u.strip()) for u in result.stdout.strip().split('\\n') if u.strip()]
            return all(u == 0 for u in utilizations)
        except Exception as e:
            print(f"Error checking GPU utilization: {{e}}")
            return False

    instance_id = get_my_instance_id()
    if not instance_id:
        print("Self-identification failed. Auto-termination daemon exiting.")
        exit(1)

    print(f"Auto-termination daemon started for instance {{instance_id}}.")
    idle_start_time = None

    while True:
        if check_gpu_idle():
            if idle_start_time is None:
                idle_start_time = time.time()
            elif (time.time() - idle_start_time) >= IDLE_TIMEOUT_SECONDS:
                print(f"GPU idle for {{IDLE_TIMEOUT_SECONDS}}s. Terminating instance...")
                try:
                    headers = {{"Authorization": f"Bearer {{LAMBDA_API_KEY}}"}}
                    payload = {{"instance_ids": [instance_id]}}
                    requests.post(TERMINATE_URL, json=payload, headers=headers, timeout=10)
                    break # Terminate signal sent, stop daemon
                except Exception as e:
                    print(f"Failed to send termination request: {{e}}")
        else:
            idle_start_time = None
        
        time.sleep(POLL_INTERVAL_SECONDS)
    EOF
    python3 /usr/local/bin/auto_terminate.py &
"""
