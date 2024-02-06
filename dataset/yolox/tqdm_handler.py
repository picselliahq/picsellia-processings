import io
import logging


class TqdmLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""
    last_buf = ""  # Added to track the last buffer content

    def __init__(self, logger, level=None):
        super(TqdmLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        if self.buf and self.buf[0:4] != self.last_buf[0:4]:
            self.logger.log(self.level, f"\n{self.buf}")
            self.last_buf = self.buf
