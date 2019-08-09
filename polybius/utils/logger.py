import yaml
import csv
import os
from tensorboardX import SummaryWriter


class Logger:
    """This class helps logging meta data and metrics of a run. It stores all logs in files in the given folder.
    """

    def __init__(self, run_folder, print_logs=True, tensorbard=True):
        """Constructor. Configures logging methods and destination.

        Args:
            run_folder (str): path to folder where log files will be saved to
            print_logs (bool): wether logs should be printed to console
            tensorboard (bool): wether a tensorboard event log should be created
        """
        self._writer = SummaryWriter(run_folder)
        self._run_folder = run_folder
        self._print_logs = print_logs
        self._tensorboard = tensorbard

        if not os.path.exists(run_folder):
            os.makedirs(run_folder)

        pass

    def _print_args(self, args):
        s = '\nArguments\n'
        s += '---------------------------------------\n'
        for key, value in vars(args).items():
            s += f'{key}: {value}\n'
        s += '---------------------------------------\n'
        print(s)

    def log_args(self, args):
        """Log the arguments/hyper parameters of the run.

        Args:
            args (argparse.namespace): Arguments used to configure the run.
        """
        if self._print_logs:
            self._print_args(args)

        args_path = os.path.join(self._run_folder, 'args.yaml')
        yaml.dump(vars(args), open(args_path, 'w'))

    def _print_metrics(self, iteration, metrics):
        s = f'{iteration}'
        for key, value in metrics.items():
            s += f' {key} : {value:.3f}'
        print(s)

    def log_metrics(self, iteration, metrics):
        """Log the metrics for one iteration.

        Args:
            iteration (int): Current iteration/step.
            metrics (dict): Keys are the names of metrics and values are the measure value for each.
        """
        if self._print_logs:
            self._print_metrics(iteration, metrics)

        if self._tensorboard:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, global_step=iteration)

        csv_path = os.path.join(self._run_folder, 'metrics.csv')
        with open(csv_path, "a") as f:
            wr = csv.writer(f)
            # Add header line if file is empty
            if os.path.getsize(csv_path) == 0:
                keys = ['iteration']
                keys.extend(metrics.keys())
                wr.writerow(keys)

            values = [iteration]
            values.extend(metrics.values())
            wr.writerow(values)
