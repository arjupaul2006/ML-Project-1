# All exception are handled here
import sys

def error_msg_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename

    error_msg = 'Error occurred in Python script name [{0}] line number [{1}]  error message [{2}]'.format(
        filename, exc_tb.tb_lineno, str(error)
    )

    return error_msg

# inherits from Exception class. Without it, raise CustomException() will throw a error.
class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_msg_detail(error_message, error_detail=error_details)

    def __str__(self):
        return self.error_message