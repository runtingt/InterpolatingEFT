"""
Handles logging tqdm output to file
"""

import io
import logging
from typing import Union, Optional

class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self, logger: logging.RootLogger, 
                 level: Optional[int]=None) -> None:
        """
        Initialise the handler

        Args:
            logger (logging.RootLogger): The logger to attach to
            level (Union[int, str], optional): The logging level to use.\
                Defaults to None.
        """
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf: str) -> int:
        """
        Write to the buffer

        Args:
            buf (str): The buffer to write to
        """
        self.buf = buf.strip('\r\n\t ')
        return 0

    def flush(self) -> None:
        """
        Flush the stream
        """
        assert self.logger
        assert self.level
        self.logger.log(self.level, self.buf)
