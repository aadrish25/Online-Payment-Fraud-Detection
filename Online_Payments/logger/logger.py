import os
import logging
from datetime import datetime

LOG_FOLDER_1=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}"

log_file_path=os.path.join("logs",LOG_FOLDER_1)
os.makedirs(log_file_path,exist_ok=True)

LOG_FILE_NAME=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
LOG_FILE_PATH=os.path.join(log_file_path,LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


