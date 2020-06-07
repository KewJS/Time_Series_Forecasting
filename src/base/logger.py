import sys, logging
import logging.config
from datetime import datetime,timedelta

'''
CRITICAL 50
ERROR	 40
WARNING	 30
INFO	 20
DEBUG	 10
OUT      9
END      8
START    7
NOTSET 	 0
'''

current_time = datetime.now()

OUT = 9 
logging.addLevelName(OUT, "OUT")
def output(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(OUT):
        self._log(OUT, message, args, **kws) 
logging.Logger.output = output


END = 8
logging.addLevelName(END, "END")
def end(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(END):
        self._log(END, message, args, **kws) 
logging.Logger.end = end


START = 7
logging.addLevelName(START, "START")
def start(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(START):
        self._log(START, message, args, **kws) 
logging.Logger.start = start


formatter = logging.Formatter(fmt='%(asctime)s| %(levelname)-8s| %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
class MyFormatter(logging.Formatter):
    datefmt='%Y-%m-%d %H:%M:%S'
    FORMATS = {
           9 : logging._STYLES['{'][0]("{message}"),
           logging.DEBUG : logging._STYLES['{'][0]("{levelname}-{module}: {message}"),
           'DEFAULT' : logging._STYLES['{'][0]("{asctime}|{levelname}: {message}")}


    def format(self, record):
        datefmt='%Y-%m-%d %H:%M:%S'
        # Ugly. Should be better
        self._style = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)

class errorcounted(object):
    """Decorator to determine number of calls for a method"""

    def __init__(self,method):
        self.method = method
        self.counter = 0


    def __call__(self,*args,**kwargs):
        self.counter += 1
        return self.method(*args,**kwargs)


class ShutdownHandler(logging.Handler):
    def emit(self, record):
        logging.shutdown()
        #raise Exception("Critical Error")
        sys.exit(1)

class Logger(object):

    def __init__(self, name):
        self.logger = logging.getLogger('my_logger')
        self.logger.propagate = False


        # add standard output stream
        stream_hdlr = logging.StreamHandler(sys.stderr)
        #stream_hdlr.setFormatter(MyFormatter())
        stream_hdlr.setFormatter(formatter)
        self.logger.addHandler(stream_hdlr)


        # add output file
        file_hdlr = logging.FileHandler(name)
        #file_hdlr.setFormatter(MyFormatter())
        file_hdlr.setFormatter(formatter)
        self.logger.addHandler(file_hdlr)

        self.logger.setLevel(7)
        #logging.basicConfig(format='%(levelname)s: %(message)s', level=9)

        self.logger.error = errorcounted(self.logger.error)
        self.logger.addHandler(ShutdownHandler(level=50))


    def start(self, message):
        global current_time
        current_time =  datetime.now()
        self.logger.start(message)


    def info(self, message):
        self.logger.info(message)


    def output(self, message):
        self.logger.output(message)


    def warning(self, message):
        self.logger.warning(message)


    def error(self, message):
        self.logger.error(message)


    def critical(self, message):
        self.logger.critical(message)


    def end(self, args):
        tdelta = datetime.now() - current_time
        tdelta -= timedelta(microseconds=tdelta.microseconds)
        if hasattr(args, 'odb'):
            self.info("Commiting results to output database.") 
            args.odb.commit()  
            args.odb.close()
        self.logger.end("Finished task {} (total time spent: {})".format(args.task.title(), tdelta))

