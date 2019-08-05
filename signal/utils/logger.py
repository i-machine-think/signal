class Logger:
    def __init__(self, print_logs=True):

        self.print_logs = print_logs
        pass

    def _print_args(self, args):
        s = '\nArguments\n'
        s += '---------------------------------------\n'
        for key, value in vars(args).items():
            s += f'{key}: {value}\n'
        s += '---------------------------------------\n'
        print(s)

    def log_args(self, args):
        if self.print_logs:
            self._print_args(args)

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
