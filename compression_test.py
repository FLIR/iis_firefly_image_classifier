from decimal import Decimal

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications import MobileNet

# Compression-related imports
from aimet_common.defs import GreedySelectionParameters
from aimet_common.defs import CostMetric, CompressionScheme
from aimet_tensorflow.defs import SpatialSvdParameters, ChannelPruningParameters, ModuleCompRatioPair
from aimet_tensorflow.compress import ModelCompressor

def evaluate_model(sess: tf.Session, eval_iterations: int, use_cuda: bool) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param sess: Tensorflow session
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """

    # Evaluate model should run data through the model and return an accuracy score.
    # If the model does not have nodes to measure accuracy, they will need to be added to the graph.
    return .5

def spatial_svd_auto_mode():

    sess = tf.Session()
    # Construct graph
    with sess.graph.as_default():
        _ = MobileNet(weights=None, input_shape=(224, 224, 3))
        print(_.summary())
        init = tf.global_variables_initializer()
    sess.run(init)

    # ignore first Conv2D op
#    conv2d = sess.graph.get_operation_by_name('block1_conv1/Conv2D')
    modules_to_ignore = None #[conv2d]

    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=10,
                                              use_monotonic_fit=True,
                                              saved_eval_scores_dict=None)

    auto_params = SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                      modules_to_ignore=modules_to_ignore)
    print([n.name for n in tf.get_default_graph().as_graph_def().node])

    params = SpatialSvdParameters(input_op_names=['input_1'], output_op_names=['act_softmax/Softmax'],
                                  mode=SpatialSvdParameters.Mode.auto, params=auto_params, multiplicity=8)
    input_shape = (1, 3, 224, 224)

    # Single call to compress the model
    compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                             working_dir=str('./'),
                                                             eval_callback=evaluate_model,
                                                             eval_iterations=10,
                                                             input_shape=input_shape,
                                                             compress_scheme=CompressionScheme.spatial_svd,
                                                             cost_metric=CostMetric.mac,
                                                             parameters=params,
                                                             trainer=None)

    print(stats)    # Stats object can be pretty-printed easily

spatial_svd_auto_mode()