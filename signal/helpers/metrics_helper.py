import pickle

import torch
from tensorboardX import SummaryWriter

from metrics.rsa import representation_similarity_analysis


class MetricsHelper:
    def __init__(self, run_folder, seed):
        self._writer = SummaryWriter(run_folder + "/" + str(seed))
        self._run_folder = run_folder
        self._best_valid_acc = -1
        self._running_loss = 0.0

    def log_metrics(
        self,
        model,
        valid_meta_data,
        valid_features,
        valid_loss_meter,
        valid_acc_meter,
        valid_entropy_meter,
        valid_messages,
        hidden_sender,
        hidden_receiver,
        loss,
        i,
    ):

        self._running_loss += loss

        num_unique_messages = len(torch.unique(valid_messages, dim=0))
        valid_messages = valid_messages.cpu().numpy()

        rsa_sr, rsa_si, rsa_ri, rsa_sm, topological_similarity, pseudo_tre = representation_similarity_analysis(
            valid_features,
            valid_meta_data,
            valid_messages,
            hidden_sender,
            hidden_receiver,
            tre=True,
        )
        l_entropy = language_entropy(valid_messages)

        if self._writer is not None:
            self._writer.add_scalar("avg_loss", valid_loss_meter.avg, i)
            self._writer.add_scalar("avg_convergence", self._running_loss / (i + 1), i)
            self._writer.add_scalar("avg_acc", valid_acc_meter.avg, i)
            self._writer.add_scalar("avg_entropy", valid_entropy_meter.avg, i)
            self._writer.add_scalar("avg_unique_messages", num_unique_messages, i)
            self._writer.add_scalar("rsa_sr", rsa_sr, i)
            self._writer.add_scalar("rsa_si", rsa_si, i)
            self._writer.add_scalar("rsa_ri", rsa_ri, i)
            self._writer.add_scalar("rsa_sm", rsa_sm, i)
            self._writer.add_scalar("topological_similarity", topological_similarity, i)
            self._writer.add_scalar("pseudo_tre", pseudo_tre, i)

        if valid_acc_meter.avg > self._best_valid_acc:
            self._best_valid_acc = valid_acc_meter.avg
            torch.save(model.state_dict(), "{}/best_model".format(self._run_folder))

        metrics = {
            "loss": valid_loss_meter.avg,
            "acc": valid_acc_meter.avg,
            "entropy": valid_entropy_meter.avg,
            "l_entropy": l_entropy,
            "rsa_sr": rsa_sr,
            "rsa_si": rsa_si,
            "rsa_ri": rsa_ri,
            "rsa_sm": rsa_sm,
            "pseudo_tre": pseudo_tre,
            "topological_similarity": topological_similarity,
            "num_unique_messages": num_unique_messages,
            "avg_convergence": self._running_loss / (i + 1),
        }
        # dump metrics
        pickle.dump(
            metrics, open("{}/metrics_at_{}.p".format(self._run_folder, i), "wb")
        )
