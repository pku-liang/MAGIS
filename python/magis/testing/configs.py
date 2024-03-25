import magis.transform as T


def get_configured_mutator(
    mutator_config_name="mlynar",
    max_level=4,
    height_limit=1,
    heat_policy=2,
    directions=(-1, 1),
    **kwargs
):
    ftree_mtor = T.FissionTreeMutator(
        max_level=max_level, height_limit=height_limit, heat_policy=heat_policy
    )
    factor_mtor = T.FissionFactorMutator(directions=directions)

    _CONFIGS = {
        "mlynar": lambda: T.ChainedMutator(
            ftree_mtor,
            factor_mtor,
            T.SwapRematMutator(),
            T.RewriteRuleMutator("taso,demem"),
        ),
        "sustr": lambda: T.ChainedMutator(
            ftree_mtor,
            factor_mtor,
            T.SwapRematMutator().truncate(100),
            T.RewriteRuleMutator("taso")
            .zip(T.RewriteRuleMutator("demem"))
            .truncate(100),
        ),
        "chen": lambda: T.ZippedMutator(
            ftree_mtor,
            factor_mtor,
            T.SwapRematMutator(),
            T.RewriteRuleMutator("taso"),
            T.RewriteRuleMutator("demem"),
        ).truncate(200),
        "ines": lambda: T.ChainedMutator(ftree_mtor, factor_mtor),
        "texas": lambda: T.ZippedMutator(
            T.SwapRematMutator(),
            T.RewriteRuleMutator("taso"),
            T.RewriteRuleMutator("demem"),
        ).truncate(200),
    }

    return _CONFIGS[mutator_config_name]()
