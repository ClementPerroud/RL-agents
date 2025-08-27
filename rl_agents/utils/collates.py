def do_nothing_collate(data):
    """
    As BaseReplayMemory implements __getitems__, the dataloader is able to call the experiences of the batch.
    Plus, the replay memories already output the correct format so collate_fn in not neccessary
    """
    return data