# imports
from typing import TypedDict
from multiprocessing import Process

class WorkerConfig(TypedDict):
    pass

class Worker:
    class Subclass:
        pass

    def __init__(self, config:WorkerConfig) -> None:
        pass

    def run(self):
        pass

def subprocess(worker_cls: type[Worker], config: WorkerConfig):
    worker = worker_cls(config)
    worker.run()

class Engine:

    def __init__(self) -> None:
        pass
    
    def start(self):
        p = Process(target=subprocess, args=(Worker, WorkerConfig())) # spawned, not forked
        p.start()
        p.join()
