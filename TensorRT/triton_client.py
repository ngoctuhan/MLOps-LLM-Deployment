#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import sys
import codecs
import numpy as np
import tritonclient.grpc as grpcclient
from utils.abs_cls import SingletonMeta

SERVER_TIMEOUT = 3.0
# unicode() doesn't exist on python3, for how we use it the
# corresponding function is bytes()
if sys.version_info.major == 3:
    unicode = bytes

class TritonClient(metaclass=SingletonMeta):

    def __init__(self, url, verbose = False) -> None:
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url, verbose=verbose
            )
        except Exception as e:
            print("context creation failed: " + str(e))
            
    def infer_string_client(self, model_name: str, input_name: str, output_name: str, string_var: str):
        """
        Client contains string input calling to Triton Serving
        
        Parameters
        ----------
        model_name : str
            The name of model
        input_name : str
            input name of request
        output_name : str
            output name of request
        string_var : str
            value of input
        
        Returns
        -------
        Any
            The output data from the inference request.
        """
        inputs = []
        outputs = []
        byte_array = codecs.encode(string_var, 'utf-8')
        bytes_data = [byte_array for _ in range(len(byte_array))]
        np_bytes_data = np.array(bytes_data, dtype=np.bytes_)
        np_bytes_data = np_bytes_data.reshape([1, len(byte_array)])
        inputs.append(grpcclient.InferInput(input_name, np_bytes_data.shape, "BYTES"))
        inputs[0].set_data_from_numpy(np_bytes_data)
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        results = self.triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        output_data = results.as_numpy(output_name)
        return output_data
    
    def infer_array_client(self, model_name: str, 
                           input_name: str, 
                           output_name: str, 
                           val_input: np.ndarray, 
                           input_shape=[1, 1, 64000],
                           version=-1):
        """
        Client for inference with numpy array input.

        Parameters
        ----------
        model_name : str
            The name of the model to use for inference.
        input_name : str
            The name of the input tensor for the model.
        output_name : str
            The name of the output tensor for the model.
        val_input : np.ndarray
            The input data as a numpy array.
        input_shape : list, optional
            The shape of the input tensor, by default [1, 1, 64000]
        version : int, optional
            The version of the model to use for inference, by default -1.
            If set to -1, the latest version of the model will be used.

        Returns
        -------
        np.ndarray
            The output data from the inference request.
        """
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(input_name, input_shape, "FP32"))
        # Create the data for the two input tensors. Initialize the first
        # to unique integers and the second to all ones.
        input0_data = np.expand_dims(val_input, axis=0)
        # # Initialize the data
        inputs[0].set_data_from_numpy(input0_data)
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        if version != -1:
            # Test with outputs
            results = self.triton_client.infer(
                model_name=model_name,
                model_version=str(version),
                inputs=inputs,
                outputs=outputs
            )
        else:
            # Test with outputs
            results = self.triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
        output_data = results.as_numpy(output_name)
        return output_data
    
    def infer_mixed_type(self, model_name:str, 
                         input_name_arr:str, 
                         input_name_str: str, 
                         output_name:str, 
                         val_input_arr:np.ndarray, 
                         val_input_str:str):

        inputs  = []
        outputs = []
        inputs.append(grpcclient.InferInput(input_name_arr, [1, val_input_arr.shape[0]], "FP32"))
        byte_array = codecs.encode(val_input_str, 'utf-8')
        bytes_data = [byte_array for _ in range(len(byte_array))]
        np_bytes_data = np.array(bytes_data, dtype=np.bytes_)
        np_bytes_data = np_bytes_data.reshape([1, len(byte_array)])
        inputs.append(grpcclient.InferInput(input_name_str, np_bytes_data.shape, "BYTES"))
        # Create the data for the two input tensors. Initialize the first
        # to unique integers and the second to all ones.
        input0_data = np.expand_dims(val_input_arr, axis=0)
        # # Initialize the data
        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(np_bytes_data)
        outputs.append(grpcclient.InferRequestedOutput(output_name))
        # Test with outputs
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=int(SERVER_TIMEOUT),
        )
        output_data = results.as_numpy(output_name)
        return output_data 