import os
import sys



class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message=error_message
        _,_,error_tb=error_details.exc_info()
        self.line_no=error_tb.tb_lineno
        self.filename=error_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Exception occured at file [{0}], line number [{1}] and error message [{2}].".format(
            self.filename,
            self.line_no,
            str(self.error_message)
        )
    

# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         raise CustomException(e,sys)