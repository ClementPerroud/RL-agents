def do_nothing_collate(data):
    """ Is used for ReplayMemory with DataLoader as they implement .__getitems__ that already pick and collate to experiences to one batched experience."""
    return data