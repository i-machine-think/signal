import torch
import numpy as np

from data.agent_vocab import AgentVocab
from .metadata_helper import get_shapes_metadata
from .dataloader_helper import get_shapes_features, get_shapes_dataloader

from enums.dataset_type import DatasetType

from models.shapes_receiver import ShapesReceiver
from models.shapes_sender import ShapesSender
from models.shapes_trainer import ShapesTrainer
from models.shapes_single_model import ShapesSingleModel
from models.shapes_meta_visual_module import ShapesMetaVisualModule


def get_sender_receiver(args):
    # Load Vocab
    vocab = AgentVocab(args.vocab_size)

    if args.task == "shapes" and not args.obverter_setup:
        cell_type = "lstm"
        genotype = {}
        if args.darts:
            raise Exception('test')
            # cell_type = "darts"
            # genotype = generate_genotype(num_nodes=args.num_nodes)
            # if not args.disable_print:
            #     print(genotype)
        if args.single_model:
            sender = ShapesSingleModel(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
            receiver = ShapesSingleModel(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
        else:
            sender = ShapesSender(
                args.vocab_size,
                args.max_length,
                vocab.bound_idx,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                greedy=args.greedy,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
            receiver = ShapesReceiver(
                args.vocab_size,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                cell_type=cell_type,
                genotype=genotype,
                dataset_type=args.dataset_type,
            )
    else:
        raise ValueError("Unsupported task type : {}".format(args.task))

    if args.sender_path:
        sender = torch.load(args.sender_path)
    if args.receiver_path:
        receiver = torch.load(args.receiver_path)

    if args.task == "shapes":
        meta_vocab_size = 15
    else:
        meta_vocab_size = 13

    if args.task == "shapes" and not args.obverter_setup:
        if args.freeze_sender:
            for param in sender.parameters():
                param.requires_grad = False
        else:
            s_visual_module = ShapesMetaVisualModule(
                hidden_size=sender.hidden_size, dataset_type=args.dataset_type
            )
            sender.input_module = s_visual_module
        if args.freeze_receiver:
            for param in receiver.parameters():
                param.requires_grad = False
        else:
            r_visual_module = ShapesMetaVisualModule(
                hidden_size=receiver.hidden_size,
                dataset_type=args.dataset_type,
                sender=False,
            )

            if args.single_model:
                receiver.output_module = r_visual_module
            else:
                receiver.input_module = r_visual_module

    return sender, receiver


def get_trainer(sender, receiver, args):
    extract_features = args.dataset_type == "raw"

    return ShapesTrainer(sender, receiver, extract_features=extract_features)


def get_training_data(args):
    # Load data
    train_data, valid_data, test_data = get_shapes_dataloader(
        batch_size=args.batch_size,
        k=args.k,
        debug=args.debugging,
        dataset_type=args.dataset_type)

    valid_meta_data = get_shapes_metadata(dataset=DatasetType.Valid)
    valid_features = get_shapes_features(dataset=DatasetType.Valid)

    return (train_data, valid_data, test_data, valid_meta_data, valid_features)


def get_raw_data(args, dataset=DatasetType.Valid):
    if args.task == "shapes":
        valid_raw = get_shapes_features(dataset=dataset, mode="raw")
        return valid_raw
    else:
        raise ValueError(
            "Unsupported task type for raw : {}".formate(args.task))


def save_example_images(args, filename):
    if args.save_example_batch:
        valid_raw = get_raw_data(args)
        valid_raw = valid_raw[:10]
        file_path = filename + "/example_batch.npy"
        np.save(file_path, valid_raw)