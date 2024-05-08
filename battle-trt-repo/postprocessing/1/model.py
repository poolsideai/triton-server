import json
import torch
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import triton_python_backend_utils as pb_utils

from tokenizer import Modality, Tokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args['model_config'])

        self.tokenizer = Tokenizer(Path(__file__).with_name("tokenizer.model"))

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

    def decode_flat(self, input):
        chunks = self.tokenizer.decode_structured(input)
        res = []
        need_end_code = False
        for el in chunks:
            if need_end_code:
                res.append("</code>")
                need_end_code = False
            if el.mod == Modality.CONTROL:
                res.append(f"<{escape(el.contents.replace(' ', ''))}>")
            elif el.mod == Modality.CODE:
                attr = ""
                if el.code_line_start is not None:
                    attr = f" code_line_start={el.code_line_start}>"
                res.append(f"<code{attr}>{escape(el.contents)}")
                need_end_code = True
            elif el.mod == Modality.NATURAL:
                res.append(f"<nl>{escape(el.contents)}</nl>")
            else:
                res.append(escape(el.contents))
        return "<root><ASSISTANT>" + "".join(res)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        logger = pb_utils.Logger
        for idx, request in enumerate(requests):
            # Get input tensors
            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, 'TOKENS_BATCH').as_numpy()

            # Get sequence length
            sequence_lengths = pb_utils.get_input_tensor_by_name(
                request, 'SEQUENCE_LENGTH').as_numpy()

            # Get cum log probs
            cum_log_probs = pb_utils.get_input_tensor_by_name(
                request, 'CUM_LOG_PROBS')

            # Get sequence length
            output_log_probs = pb_utils.get_input_tensor_by_name(
                request, 'OUTPUT_LOG_PROBS')

            # Get context logits
            context_logits = pb_utils.get_input_tensor_by_name(
                request, 'CONTEXT_LOGITS')

            # Get generation logits
            generation_logits = pb_utils.get_input_tensor_by_name(
                request, 'GENERATION_LOGITS')

            # Postprocessing output data.
            outputs = self._postprocessing(tokens_batch, sequence_lengths)

            """
            # Get input tensors
            output_ids = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_ID"
            ).as_numpy()
            batch_dim = output_ids.shape[0]
            if batch_dim != 1:

                err_str = (
                    "Inflight batching backend expects requests with batch size of 1."
                )
                logger.log_error(err_str)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(err_str)
                    )
                )
                continue

            # objects to create pb_utils.InferenceResponse.
            response_tensor = pb_utils.Tensor(
                "RESPONSE", np.array([self.decode_flat(output_ids[0])], dtype=object)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[response_tensor]
            )
            responses.append(inference_response)
            """

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(outputs).astype(self.output_dtype))

            outputs = []
            outputs.append(output_tensor)

            if cum_log_probs:
                out_cum_log_probs = pb_utils.Tensor('OUT_CUM_LOG_PROBS',
                                                    cum_log_probs.as_numpy())
                outputs.append(out_cum_log_probs)
            else:
                out_cum_log_probs = pb_utils.Tensor(
                    'OUT_CUM_LOG_PROBS', np.array([[0.0]], dtype=np.float32))
                outputs.append(out_cum_log_probs)

            if output_log_probs:
                out_output_log_probs = pb_utils.Tensor(
                    'OUT_OUTPUT_LOG_PROBS', output_log_probs.as_numpy())
                outputs.append(out_output_log_probs)
            else:
                out_output_log_probs = pb_utils.Tensor(
                    'OUT_OUTPUT_LOG_PROBS',
                    np.array([[[0.0]]], dtype=np.float32))
                outputs.append(out_output_log_probs)

            if context_logits:
                out_context_logits = pb_utils.Tensor('OUT_CONTEXT_LOGITS',
                                                     context_logits.as_numpy())
                outputs.append(out_context_logits)
            else:
                out_context_logits = pb_utils.Tensor(
                    'OUT_CONTEXT_LOGITS', np.array([[[0.0]]],
                                                   dtype=np.float32))
                outputs.append(out_context_logits)

            if generation_logits:
                out_generation_logits = pb_utils.Tensor(
                    'OUT_GENERATION_LOGITS', generation_logits.as_numpy())
                outputs.append(out_generation_logits)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=outputs)
            responses.append(inference_response)
        return responses

    def _postprocessing(self, tokens_batch, sequence_lengths):
        outputs = []
        for batch_idx, beam_tokens in enumerate(tokens_batch):
            for beam_idx, tokens in enumerate(beam_tokens):
                seq_len = sequence_lengths[batch_idx][beam_idx]
                output = self.decode_flat(
                    torch.tensor(tokens[:seq_len]),
                )
                outputs.append(output.encode('utf8'))
        return outputs

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
