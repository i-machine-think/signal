class Logger:
    def __init__(self, print_logs=True):

        self.print_logs = print_logs
        pass

    def _print_metrics(self, metrics):
        s = f'{metrics["iteration"]}'
        for key, value in metrics.items():
            if key is "iteration":
                continue

            s += f' {key} : {value:.3f}'
        print(s)

    def log_metrics(self, metrics):
        if self.print_logs:
            self._print_metrics(metrics)
