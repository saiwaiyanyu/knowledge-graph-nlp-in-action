#!/usr/bin/env python
#-*- coding:utf-8 -*-
#@Time  : 17/1/2 下午3:15
#@Author: wuchenglong

####################
# 日志纪录设置
####################

from io import StringIO as StringBuffer
log_capture_string = StringBuffer()
# log_capture_string.encoding = 'cp1251'

_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(levelname)s %(asctime)s %(process)d %(message)s'
        },
        'detail': {
            'format': '%(levelname)s %(asctime)s %(process)d ' + ' %(module)s.%(funcName)s line:%(lineno)d  %(message)s',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            # 'formatter': 'simple'
            'formatter': 'detail',
            'stream': log_capture_string,
        },
    'console1': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                # 'formatter': 'simple'
                'formatter': 'detail',
                # 'stream': log_capture_string,
            },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detail',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            # 'maxBytes': 1024,
            # 'backupCount': 3,
            'when': 'midnight',
            'interval': 1,
            'filename': 'log/debug.log'
        },
        'err_file': {
            'level': 'ERROR',
            'formatter': 'detail',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'interval': 1,
            'filename': 'log/error.log'
        },
        'perf': {
            'level': 'INFO',
            'formatter': 'simple',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'interval': 1,
            'filename':  'log/info.log'
        },
        'track': {
            'level': 'WARN',
            'formatter': 'simple',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'interval': 1,
            'filename': 'log/warn.log'
        },

    },
    'loggers': {
        'default': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'err_file', 'perf', 'track']
        },
        'console': {
            'handlers': ['file', 'err_file'],
            'level': 'DEBUG'
        },
        'perf': {
            'handlers': ['perf'],
            'level': 'DEBUG',
            'propagate': False
        },
        'track': {
            'handlers': ['track'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}


import logging.config
logging.config.dictConfig(_LOGGING)
logger = logging.getLogger('default')

if __name__ == "__main__":
    logger.debug("======= 测试=========")
    logger.info("======= 测试=========")
    logger.error("======= 测试=========")

    log_contents = log_capture_string.getvalue()
    log_capture_string.close()
    # print(log_contents.lower())
