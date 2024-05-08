import json
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET

import numpy as np
import triton_python_backend_utils as pb_utils

from tokenizer import Chunk, Modality, Tokenizer


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
        # Note: ignoring "add_special_tokens" in the config
        # Initialize tokenizer
        self.tokenizer = Tokenizer(Path(__file__).with_name("tokenizer.model"))
        # todo(honglu): get the poolside tokenizer pad_id
        self.tokenizer_end_id = self.tokenizer._model.eos_id()
        self.tokenizer_pad_id = self.tokenizer._model.eos_id()
        # Parse model output configs and convert Triton types to numpy types
        output_names = [
            "INPUT_ID", "REQUEST_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS",
            "OUT_END_ID", "OUT_PAD_ID"
        ]
        input_names = ["EMBEDDING_BIAS_WORDS", "EMBEDDING_BIAS_WEIGHTS"]
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_input_config_by_name(
                        model_config, input_name)['data_type']))

        for output_name in output_names:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, output_name)['data_type']))

    def encode_tokens(self, prompt, bos=True):
        tokens = self.encode_flat(prompt.decode("utf-8"))
        if bos:
            tokens = [self.tokenizer._model.eos_id()] + tokens.tolist()
        return tokens

    def to_chunks(self, el):
        if el.tag == "code":
            if len(el) != 0:
                raise ET.ParseError
            yield Chunk(
                el.text, Modality.CODE, code_line_start=int(el.get("code_line_start"))
            )
        elif el.tag == "nl":
            if len(el) != 0:
                raise ET.ParseError
            yield Chunk(el.text, Modality.NATURAL)
        else:
            if el.tag != "root":
                yield Chunk(el.tag, Modality.CONTROL)
            if el.text and el.text.strip():
                yield Chunk(el.text.strip(), Modality.MIXED)
            for child in el:
                yield from self.to_chunks(child)
                if child.tail and child.tail.strip():
                    yield Chunk(child.tail.strip(), Modality.MIXED)
            if el.tag != "root":
                yield Chunk(f"/ {el.tag}", Modality.CONTROL)

    def encode_flat(self, prompt):
        chunks = list(self.to_chunks(ET.fromstring(prompt)))
        for i, chunk in enumerate(chunks):
            if chunk.contents == "cut":
                del chunks[i:]
                break
        return self.tokenizer.encode(
            [chunk for chunk in chunks if chunk.contents != "/ ASSISTANT"]
        )

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

        def _log_error(err_str: str, responses: list):
            logger.log_error(err_str)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(err_str)
                )
            )

        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request, "QUERY").as_numpy()
            batch_dim = query.shape[0]
            if batch_dim != 1:
                err_str = (
                    "Inflight batching backend expects requests with batch size of 1."
                )
                _log_error(err_str, responses)
                continue

            if query.shape[1] != 1:
                err_str = (
                    f"Only accept one byte object per sample. Got multiple (input shape {str(query.shape)})"
                )
                _log_error(err_str)
                continue

            embedding_bias_words = pb_utils.get_input_tensor_by_name(
                request, 'EMBEDDING_BIAS_WORDS')
            if embedding_bias_words is not None:
                embedding_bias_words = embedding_bias_words.as_numpy()

            embedding_bias_weights = pb_utils.get_input_tensor_by_name(
                request, 'EMBEDDING_BIAS_WEIGHTS')
            if embedding_bias_weights is not None:
                embedding_bias_weights = embedding_bias_weights.as_numpy()

            embedding_bias = self._get_embedding_bias(
                embedding_bias_words, embedding_bias_weights,
                self.embedding_bias_weights_dtype)


            request_output_len = pb_utils.get_input_tensor_by_name(
                request, 'REQUEST_OUTPUT_LEN').as_numpy()

            bad_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'BAD_WORDS_DICT')
            if bad_words_dict is not None:
                bad_words_dict = bad_words_dict.as_numpy()

            stop_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'STOP_WORDS_DICT')
            if stop_words_dict is not None:
                stop_words_dict = stop_words_dict.as_numpy()

            bad_words = self._to_word_list_format(bad_words_dict)
            stop_words = self._to_word_list_format(stop_words_dict)

            # Take the end_id from the input tensors
            # If not specified, use tokenizer to get end_id
            end_id = pb_utils.get_input_tensor_by_name(request, 'END_ID')
            if end_id is not None:
                end_id = end_id.as_numpy()
            else:
                end_id = [[self.tokenizer_end_id]]

            # Take the pad_id from the input tensors
            # If not specified, use tokenizer to get pad_id
            pad_id = pb_utils.get_input_tensor_by_name(request, 'PAD_ID')
            if pad_id is not None:
                pad_id = pad_id.as_numpy()
            else:
                pad_id = [[self.tokenizer_pad_id]]

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_ids = np.array([self.encode_tokens(query[0, 0])], dtype=np.int32)
            input_lengths = np.array([[len(ids)] for ids in input_ids]).astype(np.int32)
            input_id_tensor = pb_utils.Tensor(
                "INPUT_ID", input_ids
            )
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                input_lengths.astype(self.request_input_len_dtype))
            request_output_len_tensor = pb_utils.Tensor(
                'REQUEST_OUTPUT_LEN', request_output_len)
            bad_words_ids_tensor = pb_utils.Tensor('BAD_WORDS_IDS', bad_words)
            stop_words_ids_tensor = pb_utils.Tensor('STOP_WORDS_IDS',
                                                    stop_words)
            embedding_bias_tensor = pb_utils.Tensor('EMBEDDING_BIAS',
                                                    embedding_bias)
            end_id_tensor = pb_utils.Tensor('OUT_END_ID',
                                            np.array(end_id, dtype=np.int32))
            pad_id_tensor = pb_utils.Tensor('OUT_PAD_ID',
                                            np.array(pad_id, dtype=np.int32))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[input_id_tensor, bad_words_ids_tensor, stop_words_ids_tensor,
                request_input_len_tensor, request_output_len_tensor,
                embedding_bias_tensor, end_id_tensor, pad_id_tensor]
            )
            responses.append(inference_response)
        return responses

    def _to_word_list_format(self, word_lists: List[List[str | bytes]]):
        '''
        word_lists format:
            len(word_lists) == batch_size
            word_lists[i] means the words associated to batch item i. A "word" may actually be any string. Like "lorem" or "lorem ipsum".
        '''
        assert self.tokenizer != None, "need to set tokenizer"

        if word_lists is None:
            # Return an empty array of shape (1,2,0)
            return np.empty([1, 2, 0], dtype="int32")

        flat_ids = []
        offsets = []
        for word_list in word_lists:
            item_flat_ids = []
            item_offsets = []

            for word in word_list:
                if isinstance(word, bytes):
                    word = word.decode()

                ids = self.tokenizer.encode([Chunk(word, Modality.MIXED)]).tolist()

                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)),
                                 constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)),
                                constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose(
            (1, 0, 2))

    def _get_embedding_bias(self, embedding_bias_words, embedding_bias_weights,
                            bias_dtype):

        assert self.tokenizer != None, "need to set tokenizer"

        if embedding_bias_words is None or embedding_bias_weights is None:
            return np.empty([1, 0], dtype=self.embedding_bias_weights_dtype)

        batch_embedding_bias = []
        for words, weights in zip(embedding_bias_words,
                                  embedding_bias_weights):

            vocab_size = self.tokenizer.vocab_size
            embedding_bias = [0.] * vocab_size

            assert len(words) == len(
                weights
            ), "Embedding bias words must have same dimension as embedding bias weights"

            for word, weight in zip(words, weights):
                if isinstance(word, bytes):
                    word = word.decode()
                ids = self.tokenizer.encode([Chunk(word, Modality.MIXED)])

                if len(ids) == 0:
                    continue

                for id in ids:
                    embedding_bias[id] += weight

            batch_embedding_bias.append(np.array(embedding_bias))

        return np.array(batch_embedding_bias, dtype=bias_dtype)

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
