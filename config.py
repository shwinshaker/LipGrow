

   # print information
    # move to config
    print("     ----------------------------- %s ----------------------------------" % args.arch)
    print("     depth: %i" % args.depth)
    print(model)
    print("     ----------------------------------------------------------------------")
    print("     dataset: %s" % args.dataset)

    print("     --------------------------- hypers ----------------------------------")
    print("     Epochs: %i" % args.epochs)
    print("     Train batch size: %i" % args.train_batch)
    print("     Test batch size: %i" % args.test_batch)
    print("     Learning rate: %g" % args.lr)
    print("     Momentum: %g" % args.momentum)
    print("     Weight decay: %g" % args.weight_decay)
    print("     Learning rate scheduler: %s" % args.scheduler)  # 'multi-step cosine annealing schedule'
    if args.scheduler in ['step', 'cosine', 'adacosine']:
        print("     Learning rate schedule - milestones: ", args.schedule)
    if args.scheduler in ['step', 'expo', 'adapt']:
        print("     Learning rate decay factor: %g" % args.gamma)
    if args.regularization:
        print("     Regularization: %s" % args.regularization)
        print("     Regularization coefficient: %g" % args.r_gamma)
    print("     gpu id: %s" % args.gpu_id)
    print("     num workers: %i" % args.workers)
    print("     hooker: ", args.hooker)
    print("     trace: ", args.trace)
    print("     --------------------------- model ----------------------------------")
    print("     Model: %s" % args.arch)
    print("     depth: %i" % args.depth)
    print("     block: %s" % args.block_name)
    if args.grow:
        if not args.arch in ['resnet', 'transresnet', 'preresnet']:
            raise KeyError("model not supported for growing yet.")
        print("     --------------------------- growth ----------------------------------")
        print("     grow mode: %s" % args.mode)
        print("     grow atom: %s" % args.grow_atom)
        print("     grow operation: %s" % args.grow_operation)
        print("     stepsize scaled residual: %s" % args.scale_stepsize)
        if args.mode == 'fixed':
            print("     grow milestones: ", args.grow_epoch)
        else:
            print("     max depth: %i" % args.max_depth)
            print("     scaled down err: %s" % args.scale)
            print("     err atom: %s" % args.err_atom)
            print("     err threshold: %g" % args.threshold)
            print("     smoothing scope: %i" % args.window)
            print("     reserved epochs: %i" % args.reserve)
            print("     err back track history (deprecated): %i" % args.backtrack)
    if args.debug_batch_size:
        print("     -------------------------- debug ------------------------------------")
        print("     debug batches: %i" % args.debug_batch_size)
    print("     ---------------------------------------------------------------------")