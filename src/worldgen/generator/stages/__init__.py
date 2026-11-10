"""The five stages of the pipeline, in order: elevation, erosion, hydrology, climate, biomes.

Each stage reads what the stages before it made and draws its random numbers from its own stream
(`np.random.default_rng([seed, STAGE])`), so tuning one stage never reshuffles another's.
"""
