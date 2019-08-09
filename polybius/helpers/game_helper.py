import torch
import numpy as np

from ..data.agent_vocab import AgentVocab
from .metadata_helper import get_shapes_metadata, get_metadata_properties
from .dataloader_helper import get_shapes_features, get_shapes_dataloader

from ..enums.dataset_type import DatasetType

from ..models.receiver import Receiver
from ..models.sender import Sender
from ..models.full_model import FullModel


def get_sender_receiver(device, args) -> (Sender, Receiver):
    # Load Vocab
    vocab = AgentVocab(args.vocab_size)

    receiver = None
    diagnostic_receiver = None

    cell_type = "lstm"
    # @TODO!!! ADD single model

    sender = Sender(
        args.vocab_size,
        args.max_length,
        vocab.bound_idx,
        device,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        greedy=args.greedy,
        cell_type=cell_type,
        tau=args.tau,
        vqvae=args.vqvae,
        beta=args.beta,
        discrete_latent_number=args.discrete_latent_number,
        discrete_latent_dimension=args.discrete_latent_dimension,
        discrete_communication=args.discrete_communication,
        gumbel_softmax=args.gumbel_softmax,
        rl=args.rl,
    )

    receiver = Receiver(
        args.vocab_size,
        device,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        cell_type=cell_type,
    )

    if args.sender_path:
        sender = torch.load(args.sender_path)
    if args.receiver_path:
        receiver = torch.load(args.receiver_path)

    return sender, receiver, None


def get_trainer(
    sender,
    device,
    dataset_type,
    receiver=None,
    diagnostic_receiver=None,
    vqvae=False,
    rl=False,
    entropy_coefficient=1.0,
    myopic=False,
    myopic_coefficient=0.1,
):
    extract_features = dataset_type == "raw"

    return FullModel(
        sender,
        device,
        receiver=receiver,
        diagnostic_receiver=diagnostic_receiver,
        extract_features=extract_features,
        vqvae=vqvae,
        rl=rl,
        myopic=myopic,
        myopic_coefficient=myopic_coefficient,
    )


def get_meta_data():
    train_meta_data = get_metadata_properties(dataset=DatasetType.Train)
    valid_meta_data = get_metadata_properties(dataset=DatasetType.Valid)
    test_meta_data = get_metadata_properties(dataset=DatasetType.Test)
    return train_meta_data, valid_meta_data, test_meta_data


def get_training_data(device, batch_size, k, debugging, dataset_type):
    # Load data
    train_data, valid_data, test_data = get_shapes_dataloader(
        device=device,
        batch_size=batch_size,
        k=k,
        debug=debugging,
        dataset_type=dataset_type,
    )

    valid_meta_data = get_shapes_metadata(dataset=DatasetType.Valid)
    if dataset_type != "meta":
        valid_features = get_shapes_features(device=device, dataset=DatasetType.Valid)
    else:
        valid_features = valid_meta_data

    return (train_data, valid_data, test_data, valid_meta_data, valid_features)


def get_raw_data(args, dataset=DatasetType.Valid):
    if args.task == "shapes":
        valid_raw = get_shapes_features(dataset=dataset, mode="raw")
        return valid_raw
    else:
        raise ValueError("Unsupported task type for raw : {}".formate(args.task))


def save_example_images(args, filename):
    if args.save_example_batch:
        valid_raw = get_raw_data(args)
        valid_raw = valid_raw[:10]
        file_path = filename + "/example_batch.npy"
        np.save(file_path, valid_raw)
