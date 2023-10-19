import logging

# 配置日志，将日志消息写入到一个 `.log` 文件
logging.basicConfig(filename='my_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 获取Logger
logger = logging.getLogger('my_logger')

# 记录日志
logger.info('这是一个信息消息')
logger.warning('这是一个警告消息')
logger.error('这是一个错误消息')
logger.critical('这是一个严重错误消息')
