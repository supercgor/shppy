from multiprocessing import cpu_count

def get_workers(k: int = 1) -> int:
    if k < 0:
        return cpu_count()
    return k
