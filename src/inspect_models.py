import collections

import tensorflow as tf

from model import Model, DecoderType
from main import char_list_from_file

tf.compat.v1.disable_eager_execution()


def summarize_model(arch: str) -> None:
    """Print total params, approx layer count, and variable list for a model arch."""
    tf.compat.v1.reset_default_graph()
    print(f"\n==== Inspecting architecture: {arch} ====")

    # Build graph
    _ = Model(char_list_from_file(), DecoderType.BestPath, arch=arch)

    vars_ = tf.compat.v1.trainable_variables()
    total_params = sum(v.get_shape().num_elements() for v in vars_)

    # Group parameters by high-level scope (first 1-2 components)
    layer_param_counts = collections.Counter()
    for v in vars_:
        scope_parts = v.name.split('/')[:2]
        layer_name = '/'.join(scope_parts)
        layer_param_counts[layer_name] += v.get_shape().num_elements()

    print(f"Total trainable parameters ({arch}): {total_params}")
    print(f"Approx. number of layers (unique scopes): {len(layer_param_counts)}\n")

    print("Layers summary (scope -> params):")
    for layer, n_params in sorted(layer_param_counts.items()):
        print(f"  {layer:40s} {n_params:10d}")

    print("\nTrainable variables (full list):")
    for v in vars_:
        shape = v.shape.as_list()
        print(f"  {v.name:60s} {shape}")


def main():
    summarize_model('blstm')
    summarize_model('transformer')


if __name__ == '__main__':
    main()