
import aconfig

from pdl.pdl_ast import FunctionBlock, CallBlock, Program
from pdl.pdl import parse_str, parse_file, exec_program

from granite_io.types import ChatCompletionInputs

from .base import ChatCompletionResults, InputOutputProcessor

class PdlInputOutputProcessor(InputOutputProcessor):
    """
    Base class for input-output processors that work by executing a
    Prompt Declaration Language (PDL) function processing `ChatCompletionInputs`.
    """

    _pdl_function: FunctionBlock | None
    """Handle on the current inference engine, required if this io processor's
    :func:`create_chat_completion()` method is going to be used"""

    def __init__(
        self, config: aconfig.Config | None = None, pdl: FunctionBlock | str | None = None, pdl_file: str | None = None
    ):
        """
        :param config: Setup config for this IO processor
        :param pdl: PDL function processing the `ChatCompletionInputs`
        """
        super().__init__(config)
        if isinstance(pdl, str):
            prog, _ = parse_str(pdl, file_name=pdl_file)
            self._pdl_function = prog.root
        elif pdl is None and pdl_file is not None:
            prog, _ = parse_file(pdl_file)
            self._pdl_function = prog.root
        else:
            self._pdl_function = pdl

    def create_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        prog = Program(CallBlock(call=self._pdl_function, args={"inputs": inputs.model_dump()}))
        results = exec_program(prog)
        return ChatCompletionResults.model_validate(results)
