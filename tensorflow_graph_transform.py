# =============================================================================
# Copyright (c) 2001-2019 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================

#!/usr/bin/python

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
import argparse

def main(input_graph_path, output_graph_path, width, height, channels, innodes, outnodes):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-input',
    #                     help='Input file path of the tensorflow retrained model to be transformed')
    # parser.add_argument('-innodes',
    #                     help='Ingoing graph node')
    # parser.add_argument('-outnodes',
    #                     help='Outgoing graph node')
    # parser.add_argument('-width',
    #                     help='Tensorflow training image width')
    # parser.add_argument('-height',
    #                     help='Tensorflow training image height')
    # parser.add_argument('-channels',
    #                     help='Tensorflow training image channels')
    # parser.add_argument('-output',
    #                     default='optimized_graph.pb',
    #                     help='Output file path of the tensorflow optimized model')
    # args = parser.parse_args()
    #
    # print("\r\n")
    # print(args)
    #
    # input_graph_path = args.input
    # output_graph_path = args.output
    # width = args.width
    # height = args.height
    # channels = args.channels
    # innodes = args.innodes
    # outnodes = args.outnodes

    def load_graph(graph_pb_path):
        with open(graph_pb_path, 'rb') as f:
            content = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(content)
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                return graph

    # Convert graph to graph_def
    input_graph = load_graph(input_graph_path)
    input_graph_def = input_graph.as_graph_def()

    ''' Setup the graph transform
    strip_unused_nodes:
        type: Default type for any new Placeholder nodes generated, for example int32, float, quint8.
        shape: Default shape for any new Placeholder nodes generated, as comma-separated dimensions.
               For example shape="1,299,299,3". The double quotes are important, since otherwise the commas will be
               taken as argument separators.
        name: Identifier for the placeholder arguments.
        type_for_name: What type to use for the previously-given name.
        shape_for_name: What shape to use for the previously-given name.
        https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#strip_unused_nodes
    remove_nodes:
        op: The name of the op you want to remove. Can be repeated to remove multiple ops.
        https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#remove_nodes
    fold_batch_norms:
        https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#fold_batch_norms
    fold_old_batch_norms:
        https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#fold_old_batch_norms
    '''

    strip_unused_nodes = 'strip_unused_nodes(type=float, shape="1,%s,%s,%s")' % (width, height, channels)
    remove_nodes = 'remove_nodes(op=Identity, op=CheckNumerics, op=PlaceholderWithDefault)'
    fold_batch_norms = 'fold_batch_norms'
    fold_old_batch_norms = 'fold_old_batch_norms'
    transforms = strip_unused_nodes + ' ' + remove_nodes + ' ' + fold_batch_norms + ' ' + fold_old_batch_norms

    print('Transforms parameters: ["%s"]' % transforms)
    print('Input nodes: ["%s"]' % innodes)
    print('Outgoing nodes: ["%s"]' % outnodes)

    transformed_graph_def = TransformGraph(input_graph_def,
                                           [innodes],
                                           [outnodes],
                                           [transforms])

    # Save the stripped down graph
    tf.io.write_graph(transformed_graph_def,
                      '',
                      output_graph_path,
                      as_text=False)

if __name__ == "__main__":
    main()
