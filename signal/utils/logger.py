import yaml
import os


class Logger:
    def __init__(self, run_folder, print_logs=True):

        self._run_folder = run_folder
        self._print_logs = print_logs
        pass

    def _print_args(self, args):
        s = '\nArguments\n'
        s += '---------------------------------------\n'
        for key, value in vars(args).items():
            s += f'{key}: {value}\n'
        s += '---------------------------------------\n'
        print(s)

    def log_args(self, args):
        if self._print_logs:
            self._print_args(args)

        args_path = os.path.join(self._run_folder, 'args.yaml')
        yaml.dump(vars(args), open(args_path, 'w'))

    def _print_metrics(self, metrics):
        s = f'{metrics["iteration"]}'
        for key, value in metrics.items():
            if key is "iteration":
                continue

            s += f' {key} : {value:.3f}'
        print(s)

    def log_metrics(self, metrics):
        if self._print_logs:
            self._print_metrics(metrics)
