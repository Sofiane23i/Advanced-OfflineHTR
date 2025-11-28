import tensorflow as tf
import sys

print("=" * 60)
print("TensorFlow GPU Check")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nNumber of GPUs available: {len(gpus)}")

if gpus:
    print("\nGPU Details:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    print("\n✓ GPU IS AVAILABLE - Code will use GPU")
else:
    print("\n✗ NO GPU DETECTED - Code will use CPU only")

# Check if GPU is actually being used
print("\nTesting GPU usage:")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("  GPU computation test: SUCCESS")
except RuntimeError as e:
    print(f"  GPU computation test: FAILED - {e}")

print("=" * 60)
