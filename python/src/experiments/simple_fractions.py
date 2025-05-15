#!/usr/bin/env python

# from src.utils.ray_init import init_ray

import ray
import random
import time

# From here: https://docs.ray.io/en/latest/ray-core/examples/highly_parallel.html

@ray.remote
def pi4_sample(sample_count):
    """
	pi4_sample runs sample_count experiments, and returns the 
    fraction of time it was inside the circle. 
    """
    in_count = 0
    for i in range(sample_count):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            in_count += 1
    return Fraction(in_count, sample_count)

def main():
	ray.init(f"ray://10.0.2.221:10001")

	SAMPLE_COUNT = 10000 * 10000
	start = time.time() 
	future = pi4_sample.remote(sample_count = SAMPLE_COUNT)
	pi4 = ray.get(future)
	end = time.time()
	dur = end - start
	print(f'Running {SAMPLE_COUNT} tests took {dur} seconds')

if __name__ == "__main__":
	main()
