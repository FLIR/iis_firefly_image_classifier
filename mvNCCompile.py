#! /usr/bin/env python3

# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import os
import sys
import argparse
import numpy as np

if sys.version_info[0] != 3:
    sys.stdout.write("Attempting to run with a version of Python != 3.x\n")
    sys.exit(1)

from Controllers.EnumController import *
from Controllers.FileIO import *
from Models.Blob import *
from Models.EnumDeclarations import *
from Models.MyriadParam import *
import Controllers.Globals as GLOBALS

from Controllers.Scheduler import load_myriad_config, load_network
from Controllers.PingPong import ppInit

major_version = np.uint32(2)
release_number = np.uint32(0)


def parse_args():
    parser = argparse.ArgumentParser(description="mvNCCompile.py converts Caffe or Tensorflow networks to graph files\n" +
                                     "that can be used by the Movidius Neural Compute Platform API")
    parser.add_argument('network', type=str, help='Network file (.prototxt, .meta, .pb, .protobuf)')
    parser.add_argument('-w', dest='weights', type=str, help='Weights file (override default same name of .protobuf)')
    parser.add_argument('-in', dest='inputnode', type=str, help='Input node name')
    parser.add_argument('-on', dest='outputnode', type=str, help='Output node name')
    parser.add_argument('-o', dest='outfile', type=str, default='graph', help='Generated graph file (default graph)')
    parser.add_argument('-s', dest='nshaves', type=int, default=1, help='Number of shaves (default 1)')
    parser.add_argument('-is', dest='inputsize', nargs=2, type=int, help='Input size for networks that don\'t provide an input shape, width and height expected')
    parser.add_argument('-ec', dest='explicit_concat', action='store_true', help='Force explicit concat')
    parser.add_argument('--accuracy_adjust', type=str, const="ALL:256", default="ALL:1", help='Scale the output by this amount', nargs='?')
    parser.add_argument('--ma2480', action="store_true", help="Dev flag")
    parser.add_argument('--scheduler', action="store", help="Dev flag")
    parser.add_argument('--new-parser', action="store_true", help="Dev flag")
    parser.add_argument('-i', dest='image', type=str, default='Debug', help='Image to process')
    parser.add_argument('-S', dest='scale', type=float, help='Scale the input by this amount, before mean')
    parser.add_argument('-M', dest='mean', type=str, help='Numpy file or constant to subtract from the image, after scaling')
    args = parser.parse_args()
    return args


class Arguments:

    def __init__(self, network, image, inputnode, outputnode, outfile, inputsize, nshaves, weights, explicit_concat, new_parser, extargs):
        self.net_description = network
        filetype = network.split(".")[-1]
        self.parser = Parser.TensorFlow
        if filetype in ["prototxt"]:
            self.parser = Parser.Caffe
            if weights is None:
                weights = network[:-8] + 'caffemodel'
                if not os.path.isfile(weights):
                    weights = None
        self.conf_file = network[:-len(filetype)] + 'conf'
        if not os.path.isfile(self.conf_file):
            self.conf_file = None
        self.net_weights = weights
        self.input_node_name = inputnode
        self.output_node_name = outputnode
        self.input_size = inputsize
        self.number_of_shaves = nshaves
        self.image = image
        self.raw_scale = None
        self.outputs_name = None
        self.mean = None
        self.channel_swap = None
        self.explicit_concat = explicit_concat
        self.acm = 0
        self.timer = None
        self.number_of_iterations = 2
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 1
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'generation'
        self.outputs_name = 'output'
        self.blob_name = outfile
        self.save_input = None
        self.save_output = None
        self.device_no = None
        self.new_parser = new_parser
        self.seed = -1
        self.accuracy_table = {}
        if extargs.accuracy_adjust != "":
            pairs = extargs.accuracy_adjust.split(',')
            for pair in pairs:
                layer, value = pair.split(':')
                self.accuracy_table[layer] = float(value)
        if extargs is not None:
            if hasattr(extargs, 'mean') and extargs.mean is not None:
                self.mean = extargs.mean
            if hasattr(extargs, 'scale') and extargs.scale is not None:
                self.raw_scale = extargs.scale


def create_graph(network, image, inputnode=None, outputnode=None, outfile='graph', nshaves=1, inputsize=None, weights=None, explicit_concat=None, ma2480=None, scheduler=True, new_parser=False, extargs=None):
    file_init()
    args = Arguments(network, image, inputnode, outputnode, outfile, inputsize, nshaves, weights, explicit_concat, new_parser, extargs)
    args.ma2480 = ma2480
    args.scheduler = scheduler
    GLOBALS.USING_MA2480 = args.ma2480
    GLOBALS.OPT_SCHEDULER = args.scheduler is None
    ppInit(args.scheduler)

    myriad_config = load_myriad_config(args.number_of_shaves)

    if args.conf_file is not None:
        get_myriad_info(args, myriad_config)
    filetype = network.split(".")[-1]
    parser = None
    if filetype in ["prototxt"]:
        parser = Parser.Caffe
    elif filetype in ["pb", "protobuf", "meta"]:
        parser = Parser.TensorFlow
    else:
        throw_error(ErrorTable.ParserNotSupported)

    file_gen = True
    load_ret = load_network(args, parser, myriad_config)
    net = load_ret['network']

    if args.new_parser:
        graph_file = load_ret['graph']
    else:
        graph_file = Blob([GLOBALS.BLOB_MAJOR_VERSION, GLOBALS.BLOB_MINOR_VERSION, GLOBALS.BLOB_PATCH_VERSION], net.name, '', myriad_config, net, outfile)
        graph_file.generate_v2(args)

    expected = load_ret['expected']
    if file_gen:
        np.save(args.outputs_name + "_expected.npy", expected.astype(dtype=np.float16))

def main():
    setup_warnings()

    print("mmvNCCompile v" + (u"{0:02d}".format(major_version, )) + "." +
          (u"{0:02d}".format(release_number, )) +
          ", Copyright @ Intel Corporation 2017\n") #ToDO: Clean this up, the custom formatting isn't working anyway
    args = parse_args()

    create_graph(args.network, args.image, args.inputnode, args.outputnode, args.outfile, args.nshaves, args.inputsize, args.weights, args.explicit_concat, args.ma2480, args.scheduler, args.new_parser, args)

if __name__ == "__main__":
    main()
