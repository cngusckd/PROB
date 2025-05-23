try:
    from resource import getrusage, RUSAGE_CHILDREN, RUSAGE_SELF

    def get_memory_mb():
        """
        Get the memory usage of the current process and its children.

        Returns:
            dict: A dictionary containing the memory usage of the current process and its children.

            The dictionary has the following keys:
                - self: The memory usage of the current process.
                - children: The memory usage of the children of the current process.
                - total: The total memory usage of the current process and its children.
        """
        res = {
            "self": getrusage(RUSAGE_SELF).ru_maxrss / 1024,
            "children": getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024,
            "total": getrusage(RUSAGE_SELF).ru_maxrss / 1024 + getrusage(RUSAGE_CHILDREN).ru_maxrss / 1024
        }
        return res
except BaseException:
    get_memory_mb = None

try:
    import torch

    if torch.cuda.is_available():
        def get_alloc_memory_by_torch() -> list[int]:
            """
            Returns GPU memory allocated by the current PyTorch process.
            Values are in Bytes.
            """
            allocated = []
            for i in range(torch.cuda.device_count()):
                _ = torch.tensor([1], device=f'cuda:{i}')  # force context init
                allocated.append(torch.cuda.max_memory_allocated(i))

            return allocated

        def get_memory_gpu_mb():
            """
            Get the memory usage of all GPUs in MB.
            """

            return [d / 1024 / 1024 for d in get_alloc_memory_by_torch()]
    else:
        get_memory_gpu_mb = None
except BaseException:
    get_memory_gpu_mb = None