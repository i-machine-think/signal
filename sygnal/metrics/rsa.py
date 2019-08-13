import numpy as np
import scipy.spatial
import scipy.stats


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def representation_similarity_analysis(
    test_images,
    test_metadata,
    generated_messages,
    hidden_sender,
    hidden_receiver,
    samples=5000,
    tre=False,
):
    """
    Calculates RSA scores of the two agents (ρS/R),
    and of each agent with the input (ρS/I and ρR/I),
    and Topological Similarity between metadata/generated messages
    where S refers to Sender,R to Receiver,I to input.
    Args:
        test_set: encoded test set metadata info describing the image
        generated_messages: generated messages output from eval on test
        hidden_sender: encoded representation in sender
        hidden_receiver: encoded representation in receiver
        samples (int, optional): default 5000 - number of pairs to sample
        tre (bool, optional): default False - whether to also calculate pseudo-TRE
    @TODO move to metrics repo
    """
    # one hot encode messages by taking padding into account and transforming to one hot
    messages = one_hot(generated_messages)

    # if input is metadata
    if test_images is None:
        test_images = test_metadata

    # this is needed since some samples might have been dropped during training to maintain batch_size
    test_images = test_images[: len(messages)]
    test_metadata = test_metadata[: len(messages)]

    assert test_metadata.shape[0] == messages.shape[0]

    sim_image_features = np.zeros(samples)
    sim_metadata = np.zeros(samples)
    sim_messages = np.zeros(samples)
    sim_hidden_sender = np.zeros(samples)
    sim_hidden_receiver = np.zeros(samples)

    for i in range(samples):
        rnd = np.random.choice(len(test_metadata), 2, replace=False)
        s1, s2 = rnd[0], rnd[1]

        sim_image_features[i] = scipy.spatial.distance.cosine(
            test_images[s1], test_images[s2]
        )
        sim_metadata[i] = scipy.spatial.distance.cosine(
            test_metadata[s1], test_metadata[s2]
        )

        sim_messages[i] = scipy.spatial.distance.cosine(
            messages[s1].flatten(), messages[s2].flatten()
        )
        sim_hidden_sender[i] = scipy.spatial.distance.cosine(
            hidden_sender[s1].flatten(), hidden_sender[s2].flatten()
        )
        sim_hidden_receiver[i] = scipy.spatial.distance.cosine(
            hidden_receiver[s1].flatten(), hidden_receiver[s2].flatten()
        )

    rsa_sr = scipy.stats.pearsonr(sim_hidden_sender, sim_hidden_receiver)[0]
    rsa_si = scipy.stats.pearsonr(sim_hidden_sender, sim_image_features)[0]
    rsa_ri = scipy.stats.pearsonr(sim_hidden_receiver, sim_image_features)[0]
    # added rsa_sm to compare between message and internal of sender
    rsa_sm = scipy.stats.pearsonr(sim_hidden_sender, sim_messages)[0]

    topological_similarity = scipy.stats.pearsonr(sim_messages, sim_metadata)[0]

    if tre:
        pseudo_tre = np.linalg.norm(sim_metadata - sim_messages, ord=1)
        return rsa_sr, rsa_si, rsa_ri, rsa_sm, topological_similarity, pseudo_tre
    else:
        return rsa_sr, rsa_si, rsa_ri, rsa_sm, topological_similarity
