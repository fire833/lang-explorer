
import ray
import os

# Wrapper function to apply to ray in a dynamic way. 
# Will be useful when swapping over to schooner.
def init_ray():
	addr = os.getenv("RAY_URL")

	if addr == None:
		addr = "client.ray.soonerhpclab.org:10001"

	if not ray.is_initialized():
		ray.init(f"ray://{addr}")
