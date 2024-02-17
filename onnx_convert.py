import os
import time
import onnx
import numpy as np
from glob import glob

import tf2onnx
import onnxoptimizer
import onnxruntime as rt
import tensorflow as tf
# from onnxconverter_common import float16

import womoku as gf


# brain damage max([int(path.split("\\")[-1][1:]) for path in glob("alphazero/models/*")]),
# didn't put the : in front of the 1, so 9 was the biggest generation possible BRU
def Convert2onnx(tf_model, generation):
    print(f"Converting Generation {generation}")

    # temporary output path without optimizations
    print((None, *gf.SHAPE[1:]))
    if gf.TF_DTYPE.upper() == "INT8":
        TF_DTYPE = tf.int8
    elif gf.TF_DTYPE.upper() == "INT16":
        TF_DTYPE = tf.int16
    tf2onnx.convert.from_keras(tf_model,
                               input_signature=[tf.TensorSpec((None, *gf.SHAPE[1:]), TF_DTYPE)],
                               output_path=f'alphazero/tmp/{generation}.onnx')

    model = onnx.load(f'alphazero/tmp/{generation}.onnx')

    # optimize
    model = onnxoptimizer.optimize(model,
                                   ['nop', 'eliminate_nop_cast', 'eliminate_nop_dropout', 'eliminate_nop_flatten',
                                    'extract_constant_to_initializer', 'eliminate_if_with_const_cond',
                                    'eliminate_nop_monotone_argmax', 'eliminate_nop_pad', 'eliminate_nop_concat',
                                    'eliminate_nop_split', 'eliminate_nop_expand', 'eliminate_shape_gather',
                                    'eliminate_slice_after_shape', 'eliminate_nop_transpose',
                                    'fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'fuse_consecutive_concats',
                                    'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze',
                                    'fuse_consecutive_squeezes', 'fuse_consecutive_transposes',
                                    'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv', 'fuse_pad_into_pool',
                                    'fuse_transpose_into_gemm', 'fuse_concat_into_reshape', 'eliminate_nop_reshape',
                                    'eliminate_nop_with_unit', 'eliminate_common_subexpression', 'fuse_qkv',
                                    'fuse_consecutive_unsqueezes', 'eliminate_deadend', 'eliminate_identity',
                                    'eliminate_shape_op', 'fuse_consecutive_slices', 'eliminate_unused_initializer',
                                    'eliminate_duplicate_initializer'])

    with open(f"alphazero/onnx_models/{generation}.onnx", "wb") as f:
        f.write(model.SerializeToString())

    print(f"Saved model at alphazero/onnx_model/{generation}.onnx")
    os.remove(f'alphazero/tmp/{generation}.onnx')

    # if not gf.USE_GPU:
    #     sess_options = rt.SessionOptions()
    #     sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    #     sess_options.optimized_model_filepath = f"alphazero/onnx_optimized/{generation}.onnx"
    #     sess_options.intra_op_num_threads = 2
    #     sess_options.inter_op_num_threads = 1
    #     session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
    #                                   providers=gf.PROVIDERS, sess_options=sess_options)
    #     input_name = session.get_inputs()[0].name
    #     data = np.random.randint(low=-1, high=2, size=(gf.MAX_NODES_INFER, *gf.SHAPE[1:]), dtype=np.int8)
    #     for _ in range(100):
    #         x = session.run(["policy", "value"], {input_name: data})
    #
    #     start = time.time()
    #     for _ in range(500):
    #         x = session.run(["policy", "value"], {input_name: data})
    #     time_taken = time.time() - start
    #     print(f"It took {time_taken} seconds for 500 * {gf.MAX_NODES_INFER} steps")
    #     print(f"Throughput is {(500 * gf.MAX_NODES_INFER) / time_taken} steps per second")

def Convert2onnx_GPU(tf_model, generation):
    print(f"Converting Generation {generation}")
    tf2onnx.convert.from_keras(tf_model,
                               input_signature=[tf.TensorSpec((None, *gf.SHAPE[1:]), tf.int16)],
                               output_path=f'alphazero/tmp/{generation}.onnx')

    model = onnx.load(f'alphazero/tmp/{generation}.onnx')

    # optimize
    model = onnxoptimizer.optimize(model,
                                   ['nop', 'eliminate_nop_cast', 'eliminate_nop_dropout', 'eliminate_nop_flatten',
                                    'extract_constant_to_initializer', 'eliminate_if_with_const_cond',
                                    'eliminate_nop_monotone_argmax', 'eliminate_nop_pad', 'eliminate_nop_concat',
                                    'eliminate_nop_split', 'eliminate_nop_expand', 'eliminate_shape_gather',
                                    'eliminate_slice_after_shape', 'eliminate_nop_transpose',
                                    'fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'fuse_consecutive_concats',
                                    'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze',
                                    'fuse_consecutive_squeezes', 'fuse_consecutive_transposes',
                                    'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv', 'fuse_pad_into_pool',
                                    'fuse_transpose_into_gemm', 'fuse_concat_into_reshape', 'eliminate_nop_reshape',
                                    'eliminate_nop_with_unit', 'eliminate_common_subexpression', 'fuse_qkv',
                                    'fuse_consecutive_unsqueezes', 'eliminate_deadend', 'eliminate_identity',
                                    'eliminate_shape_op', 'fuse_consecutive_slices', 'eliminate_unused_initializer',
                                    'eliminate_duplicate_initializer'])

    with open(f"alphazero/onnx_GPU/{generation}.onnx", "wb") as f:
        f.write(model.SerializeToString())

    print(f"Saved model at alphazero/onnx_model/{generation}.onnx")
    os.remove(f'alphazero/tmp/{generation}.onnx')


if __name__ == "__main__":
    from net.resnext2d import Get_2dresnext
    import womoku as gf

    # latest generation
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
    model = tf.keras.models.load_model(f"alphazero/models/{generation}", compile=False)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=2e-4),
                  loss={"policy": "bce", "value": "mse"},
                  loss_weights={"policy": 1, "value": 1},
                  metrics=["accuracy", "bce", "mse"],
                  jit_compile=False)
    Convert2onnx(model, generation)
    Convert2onnx_GPU(model, generation)

