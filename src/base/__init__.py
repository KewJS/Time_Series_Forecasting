import os, sys
from datetime import datetime
from platform import uname, system
from glob import glob
from collections import OrderedDict
from itertools import cycle
from queue import Queue, Empty
import argparse
import uuid
import pandas as pd
import numpy as py
from functools import partial
from contextlib import contextmanager

from base.logger import Logger
from base.db_connection import DataBase

__sys_version__ = sys.version_info
__date_format__ = "%Y-%m-%d %H:%M:%S"
__user__ = os.getenv('LOGNAME') or os.getlogin()
__sys_name__ = uname()[1]


mess_dict = dict(WARNING="orange", INFO="green", ERROR="red", CRITICAL="red")
def script(queues, outputs):
    for task in cycle(queues):
        if not task in outputs:
            outputs[task] = []
        # read line without blocking
        try:  
            line = queues[task].get(timeout=0.5).strip().decode('UTF-8') 
            #line = queues[task].get_nowait().strip().decode('UTF-8')
        except Empty:
            pass
        except KeyError:
            break
        else: 
            try:
                time, typ, mess = line.split('|')
                time = time.strip()
                typ = typ.strip()
                out =''
                #print(typ, mess)
                row = '<tr><td>{}</td><td><font color="{}">{}</font></td><td>{}</td></tr>'.format(time, mess_dict.get(typ, "black"), typ, mess)
                if typ == "END":
                    del queues[task]
                    out = row
                elif typ in mess_dict or typ == 'START':
                    out = row
                if out != '':
                    outputs[task].insert(0, out)
                    if typ == "END":
                        out +=  "|END"
                    #print("data:{}|{}\n\n".format(task, out))
                    yield "data:{}|{}\n\n".format(task, out)

            except ValueError:
                pass

class Base_Parse(object):

    def is_valid_file(self, parser, arg):
        if arg != '' and not os.path.exists(arg):
            parser.error("The file %s does not exist!" % arg)
        else:
            return open(arg, 'r')  # return an open file handle

    def restricted_float(self, x):
        frange = (0, 1)
        x = float(x)
        if x < frange[0] or x > frange[1]:
            raise argparse.ArgumentTypeError("%r not in range [{}, {}}]"%(frange))
        return x

    
    def get_vars(self, ignore_list=[]):
        tasks = self.parser._get_positional_actions()[0].choices
        options = {}
        for key, task in tasks.items():
            options[key] = OrderedDict()
            for action in task._actions:
                pars = vars(action)
                if pars['dest'] not in ['help', 'version']+ignore_list:
                    options[key][pars['dest']] = {p:pars[p] for p in pars if not p in ["dest", 'metavar', 'container']}
                    options[key][pars['dest']]['general'] = False
        # general options
        for task in options:
            for action in self.parser._actions:
                if type(action) in [argparse._StoreTrueAction, argparse._StoreAction]:
                    pars = vars(action)
                    if not pars['dest'] in ignore_list:
                        options[task][pars['dest']] = {p:pars[p] for p in pars if not p in ["dest", 'metavar', 'container']}
                        options[task][pars['dest']]['general'] = True
        
        return options


def process_dirs(paths, logger, qtrain=True, exts=['jpg', 'png', 'jpeg']):
    """ process directory for training or test images """
    images = {}
    for path in paths:
        if path == '': 
            continue
        if not os.path.isdir(path):
            if logger != None:
                logger.warning("Directory path '{}' not found.".format(path))
            continue

        # labels?
        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))] if qtrain else ['unknown']

        # loop over the labelsand parse image files
        for label in labels:
            if qtrain:
                base_dir = os.path.join(path, label, '*')
            else:
                base_dir = os.path.join(path, '*') if os.path.exists(path) else path
            if not label in images:
                images[label] = []
            for ext in exts:
                images[label].extend(glob('{}.{}'.format(base_dir, ext)))
    
    # remove empty classes
    for label in list(images.keys()):
        if images[label] == []:
            if logger != None:
                logger.warning("no '{}' images file found in path '{}'.".format(','.join(exts), os.path.join(path, label)))
            del images[label] 
    
    return images


def process_args(args, kwargs, options):
    """ process arguments"""
    # create the output directory
    out_dir = args.output
        
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        if not ((hasattr(args, 'realtime') and args.realtime)
            or (hasattr(args, 'use_prefit_model') and args.use_prefit_model)):
            dtime = datetime.fromtimestamp(os.path.getctime( out_dir)).strftime('%y%m%d_%H%M%S')
            name =  out_dir+'_'+ dtime
            os.rename( out_dir, name)

    if not os.path.exists(out_dir):
        os.makedirs( out_dir)
    
    # create logger 
    log_name = os.path.join(out_dir, args.task.title() + '.log')
    args.logger = Logger(log_name)
    args.date_format = __date_format__

    # standard output the running of module with current Python version    
    args.logger.start("Running task '{task}', using '{prog} v{version}' under Python {py} by '{user}' on '{sys}', output to  '{output}'".format(
            task=args.task,
            prog=os.path.split(kwargs['__current_module__'])[1], 
            version=kwargs['__version__'], 
            py='.'.join(map(str, __sys_version__[:3])), 
            user=__user__, 
            sys=__sys_name__,
            output=args.output)
    )
    #args.logger.info("using arguments: {args}".format(args=args.__dict__))

    if hasattr(args, 'force_cpu') and args.force_cpu:
        # force tensoflow to use CPU only on machines with a GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  

    # establish the input database connection if required
    if hasattr(args, 'idb_connstr') and args.idb_connstr != '':
        try:
            # using sqlalchemy
            #iengine = create_engine(conn_str=args.idb_connstr)
            #args.idb = iengine.connect()

            # using turbodbc
            args.idb = DataBase(conn_str=args.idb_connstr)

        except:
            args.logger.critical("failed to connect to the input Database using the connection strings entered with '-d'.")

    # establish the input database connection if required
    if hasattr(args, 'odb_connstr') and args.odb_connstr != '':
        try:
            engine = create_engine(args.odb_connstr, convert_unicode=True)
            args.odb = Session(engine)
            try:
                __, abbr, project_name, run_name, task = args.output.split(os.sep)
                args.logger.info("storing run '%s' records in the database." % run_name)
                product = args.odb.query(Product).filter_by(abbr=abbr).first()
                if not product:
                    args.logger.error("product abbr '%s' is not valid." % abbr)
                    raise ValueError("product abbr '%s' is not valid." % abbr)
                project = args.odb.query(Project).filter_by(name=project_name, product=product).first()
                if not project:
                    args.logger.error("project name '%s' is not valid." % project_name)
                    raise ValueError("project name '%s' is not valid." % project_name)
                user = args.odb.query(User).filter_by(login=args.user).first()
            except:
                args.logger.error("failed to parse the project properties from the '-o' argument; the run won't be recorded in the database.")
                raise ValueError("failed to parse the project properties from the '-o' argument; the run won't be recorded in the database.")
            
            args.logger.info("storing run '%s' records in the database." % run_name)
            run = args.odb.query(Run).filter_by(name=run_name, project=project).first()
            if run:                
                args.logger.warning("run '%s' already exists; replacing the run results.")
                args.odb.delete(run)
                args.odb.flush()
            run = Run(name=run_name, project=project, project_id=project.id, task=task)
            project.runs.append(run)
            if user:
                run.user = user
            inputs = []
            for k in args.__dict__:
                if getattr(args, k) is None:
                    pass
                elif k in ['idb', 'odb', 'idb_connstr', 'odb_connstr', 'logger', 'date_format', 'task', 'output']:
                    pass
                elif getattr(args, k) != '':
                    inputs.append(Parameter(name=k, value=str(getattr(args, k))))
            run.inputs = inputs
            args.run = run
            args.odb.add(run)
            args.odb.flush()            
        except:
            args.logger.critical("failed to connect to and store run records in the output Database.")
    
    if not hasattr(args, 'show'):
        setattr(args, 'show', False)

    if not args.show:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
        # suppress warning and debug info

    # process list-type options
    for key, option in options.items():
        if option['nargs'] == '+' and type(getattr(args, key)) != list:
            setattr(args, key, [getattr(args, key)])
        
    # check the range for 'Test-to-Train' ratio
    if hasattr(args, 'test_ratio'):
        frange = (0, 1)
        if args.test_ratio < frange[0] or args.test_ratio > frange[1]:
            args.logger.critical("'test_ratio' is not in range [{}, {}}]".format(frange))


def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


def convert_input(X):
    """
    Unite data into a DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, list):
            X = pd.DataFrame(np.array(X))
        elif isinstance(X, (np.generic, np.ndarray)):
            X = pd.DataFrame(X)
        elif isinstance(X, csr_matrix):
            X = pd.DataFrame(X.todense())
        else:
            raise ValueError('Unexpected input type: %s' % (str(type(X))))

        X = X.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    return X


def get_generated_cols(X_original, X_transformed, to_transform):
    """
    Returns a list of the generated/transformed columns.

    Arguments:
        X_original: df
            the original (input) DataFrame.
        X_transformed: df
            the transformed (current) DataFrame.
        to_transform: [str]
            a list of columns that were transformed (as in the original DataFrame), commonly self.cols.

    Output:
        a list of columns that were transformed (as in the current DataFrame).
    """
    original_cols = set(X_original.columns)
    current_cols = set(X_transformed.columns)
    generated_cols = list(current_cols - (original_cols - set(to_transform)))

    return generated_cols            