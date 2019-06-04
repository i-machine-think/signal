import torch.nn as nn

class ShapesMetaVisualModule(nn.Module):
    def __init__(
        self,
        features_dim=512,
        meta_dim=15,
        hidden_size=512,
        dataset_type="meta",
        sender=True,
    ):
        super(ShapesMetaVisualModule, self).__init__()
        self.dataset_type = dataset_type
        self.features_dim = features_dim
        self.hidden_size = hidden_size
        self.process = False ## TODO ?

        if dataset_type == "features":
            if features_dim == hidden_size:
                self.process = False
            else:
                self.process_input = nn.Linear(
                    *(features_dim, hidden_size)
                    if sender
                    else (hidden_size, features_dim)
                )

        if dataset_type == "meta":
            self.process_input = nn.Linear(
                *(meta_dim, hidden_size) if sender else (hidden_size, meta_dim)
            )

    def reset_parameters(self):
        if self.process:
            self.process_input.reset_parameters()

    def forward(self, input):
        # reduce features to hidden, or hidden to features
        # or reduce metadata to hidden, or hidden to metadata
        if self.process and (
            self.dataset_type == "features" or self.dataset_type == "meta"
        ):
            input = self.process_input(input)

        return input
