import numpy as np
from glob import glob
import onnxruntime as rt

import womoku as gf


def Build_trt_cache():
    # run inference 1 time in order for tensorrt engine to build the cache
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])

    sess_options = rt.SessionOptions()

    session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                  providers=gf.PROVIDERS, sess_options=sess_options)

    input_name = session.get_inputs()[0].name

    for _ in range(10):
        raw_policy, raw_value = session.run(["policy", "value"], {
            input_name: np.random.randint(low=-1, high=1, size=gf.SHAPE).astype(gf.NP_DTYPE)})

    del generation, raw_policy, raw_value, session
    print("Successfully built trt cache")


if __name__ == "__main__":
    Build_trt_cache()
