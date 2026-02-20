import os
import httpx
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ProvisionResult:
    """Result of GPU provisioning. Returned by provision_gpu()."""
    ip: str
    instance_id: str
    launched_by_us: bool  # True if we launched; False if we reused existing


class LambdaLifecycleManager:
    """Manager for Lambda Cloud GPU instances lifecycle.
    
    Automates the discovery, provisioning, and readiness checking of GPU instances.
    
    Note: Lambda Cloud does not offer spot or preemptible instances; only on-demand.
    For cost optimization, consider reserved capacity or alternative providers.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        self.base_url = "https://cloud.lambdalabs.com/api/v1"
        if not self.api_key:
            logger.warning("LAMBDA_API_KEY not found in environment or constructor.")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def get_existing_instance(self, instance_type: str = "gpu_1x_a100_sxm4") -> Optional[Dict]:
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

    async def provision_gpu(
        self,
        instance_type: str = "gpu_1x_a100_sxm4",
        region: str = "us-west-1",
        on_stage: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> Optional[ProvisionResult]:
        """Provisions a new GPU instance or returns the IP of an existing one.
        
        Args:
            instance_type: The Lambda Labs instance type name.
            region: The region name to launch in.
            on_stage: Optional async callback(stage, message) for progress reporting.
            
        Returns:
            ProvisionResult with ip, instance_id, and launched_by_us, or None if provisioning fails.
        """
        async def report(stage: str, message: str):
            if on_stage:
                await on_stage(stage, message)

        if not self.api_key:
            logger.error("Cannot provision GPU: LAMBDA_API_KEY is missing.")
            return None
            
        # 1. Detection: Find existing active or booting instance
        await report("discovering", "Checking for existing Lambda GPU instances...")
        instance = await self.get_existing_instance(instance_type)
        instance_id = None
        launched_by_us = False
        
        if instance:
            instance_id = instance.get("id")
            if instance.get("status") == "active" and instance.get("ip_address"):
                ip = instance.get("ip_address")
                logger.info(f"Using existing active instance: {instance_id} at {ip}")
                return ProvisionResult(ip=ip, instance_id=instance_id, launched_by_us=False)
            else:
                logger.info(f"Found existing instance {instance_id} with status '{instance.get('status')}'. Waiting for it to become active...")
        else:
            # 2. Launch Logic: Launch new instance if none exists
            await report("launching", f"Launching new '{instance_type}' in '{region}'...")
            logger.info(f"No active or booting instance found. Launching new '{instance_type}' in '{region}'...")
            
            # Get user data with GPU status server (no API key - backend handles termination)
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
                    launched_by_us = True
            except Exception as e:
                logger.error(f"Error during instance launch: {e}")
                return None

        # 3. Wait for IP and active status
        if not instance_id:
            return None

        await report("waiting_for_active", f"Waiting for instance '{instance_id}' to become active...")
        logger.info(f"Waiting for instance '{instance_id}' to become active...")
        max_attempts = 240  # ~20 minutes with 5s polling
        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(max_attempts):
                try:
                    resp = await client.get(f"{self.base_url}/instances/{instance_id}", headers=self.headers)
                    if resp.status_code == 200:
                        inst_data = resp.json().get("data", {})
                        if inst_data.get("status") == "active" and inst_data.get("ip_address"):
                            ip = inst_data["ip_address"]
                            logger.info(f"Instance '{instance_id}' is now active at {ip}")
                            return ProvisionResult(ip=ip, instance_id=instance_id, launched_by_us=launched_by_us)
                        elif inst_data.get("status") == "terminated":
                            logger.error(f"Instance '{instance_id}' was terminated unexpectedly.")
                            return None
                    else:
                        logger.warning(f"Failed to poll instance status (Attempt {attempt+1}): {resp.status_code}")
                except Exception as e:
                    logger.warning(f"Error polling instance status: {e}")
                
                await asyncio.sleep(5)
                        
        logger.error(f"Instance '{instance_id}' did not become active within the timeout period.")
        return None

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a Lambda Cloud instance by ID.
        
        Returns:
            True if termination was successful, False otherwise.
        """
        if not self.api_key:
            logger.error("Cannot terminate: LAMBDA_API_KEY is missing.")
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/instance-operations/terminate",
                    json={"instance_ids": [instance_id]},
                    headers=self.headers
                )
                response.raise_for_status()
                logger.info(f"Termination requested for instance {instance_id}")
                return True
        except Exception as e:
            logger.error(f"Error terminating instance {instance_id}: {e}")
            return False

    async def check_gpu_idle(self, gpu_ip: str) -> bool:
        """Check if GPU utilization is 0% on the instance.
        
        Fetches http://{gpu_ip}:9999/gpu-util which returns nvidia-smi output.
        Returns True if all GPUs report 0% utilization.
        """
        url = f"http://{gpu_ip}:9999/gpu-util"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return False
                text = response.text.strip()
                if not text:
                    return False
                utilizations = [int(u.strip()) for u in text.split("\n") if u.strip()]
                return all(u == 0 for u in utilizations)
        except Exception as e:
            logger.debug(f"GPU idle check failed for {gpu_ip}: {e}")
            return False

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
        """Generates the cloud-init user_data script for instance provisioning.
        
        No API key or credentials are embedded. The backend polls the GPU status
        endpoint and handles termination via Lambda API.
        """
        return """#cloud-config
runcmd:
  - apt-get update
  - apt-get install -y curl python3-pip
  - pip install requests

  # Install and start LM Studio CLI (lms)
  - curl -fsSL https://lmstudio.ai/install.sh | bash
  - export PATH="$PATH:$HOME/.cache/lm-studio/bin" # Ensure lms is in path
  - source ~/.bashrc
  - lms daemon up &
  - sleep 10

  # Start LM Studio server and load model
  - lms server start --port 1234 --bind 0.0.0.0
  - lms get medgemma-27b-text-it-GGUF@q4_k_m -y
  - lms load medgemma-27b-text-it -c 16384

  # GPU status server on port 9999 - backend polls this for idle detection and termination
  # No API key or credentials - backend handles termination via Lambda API
  - |
    cat <<'GPUSTATUS' > /usr/local/bin/gpu_status_server.py
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import subprocess

    class GPUUtilHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/gpu-util':
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True
                    )
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(result.stdout.encode())
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(str(e).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress access logs

    HTTPServer(("0.0.0.0", 9999), GPUUtilHandler).serve_forever()
    GPUSTATUS
    python3 /usr/local/bin/gpu_status_server.py &
"""
