def spawn(map_fn, args=None, nprocs=None, start_method='spawn'):
    map_fn(1, *args)
