import logging

def myLog(name, fname = 'myGlobalLog.log'):

    logger = logging.getLogger(name);
    logger.setLevel(logging.DEBUG)
    fhan = logging.FileHandler(fname)
    fhan.setLevel(logging.DEBUG)
    logger.addHandler(fhan)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhan.setFormatter(formatter)
    '''comment this to enable requests logger'''
    #logger.disabled = True
    return logger