class Logger:
    def __init__(self):
        pass

    def log_metrics(self, metrics):
        s = f'{metrics["iteration"]}'
        for key, value in metrics.items():
            if key is "iteration":
                continue

            s += f' {key} : {value:.3f}'
        print(s)
