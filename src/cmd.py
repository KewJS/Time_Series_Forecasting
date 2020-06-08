import os, sys, fnmatch
import time, inspect
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore')

from .base.logger import Logger
from .base import Base_Parse, process_args
# from .base.db_connection import DataBase

from .Config import Config
from .analysis import Analysis
from .train import Train


# Set module attributes
__version__ = "1.0.0"
__description__ = 'Advanced Analytics for Time Series Forecasting'
__author__ = 'kew.jingsheng@petronas.com.my'
__current_module__ = inspect.getfile(inspect.currentframe()).replace('__main__','').replace('__.py','')
__path__ = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

module_args = {k:v for k, v in globals().items() if len(k) > 4 and k[:2] == '__' and k[-2:] == '__'}


def main(argsv):
    """ main driver for performing PWM """
    try:
        #print(args)
        parse = Parse()
        args = parse.parser.parse_args(argsv)

        arg_options = super(Parse, parse).get_vars()
        
        if args.task == 'report':
            args.idb_connstr = ''

        """ 
        comment out to fix turbodbc
        """
        # else:
        #     # process input database
        #     args.idb_connstr = "DATABASE={};DRIVER=SQL Server;SERVER={SERVER};PORT={PORT};UID={UID};PWD={PWD};READONLY=True"\
        #                     .format(args.idb_name, **Config._DATABASES[args.idb_name])

        # process the arguments
        if not hasattr(args, "output"):
            args.output = Config.FILES["DATA_LOCAL"]
        process_args(args, module_args, arg_options[args.task])

        #    
        # perform 'report' task
        #
        if args.task == 'report':      

            args.logger.info("Setting Jupyter extensions ...")
            os.system("jupyter contrib nbextension install --user")
            os.system("jupyter nbextension enable python-markdown/main")
            os.system("jupyter nbextension enable --py --sys-prefix widgetsnbextension")
            os.system("jupyter nbextension enable --py --sys-prefix qgrid")
            if args.exec:
                args.logger.info("Executing Jupyter notebook and exporting to html ...")
                os.system("jupyter nbconvert --to html --no-input --template {} --ExecutePreprocessor.store_widget_state=True --execute {} ".format(args.template, args.input))
            else:
                args.logger.info("exporting Jupyter notebook as HTML file ...")
                os.system("jupyter nbconvert --to html --no-input --template {} {} ".format(args.template, args.input))
    
        #
        # Perform EDA 
        #       
        elif args.task == 'analysis':
            # Perform analysis and clean up data
            args.logger.info("Analyzing input data:")
            if args.district == [None]:
                args.district = None
            analysis = Analysis(district=args.district, suffix=args.suffix, logger=args.logger)

            args.logger.info(" Loading Sales Data ...")
            analysis.get_biodiesel()
            analysis.get_primax95()


        #
        # Perform training 
        #       
        elif args.task == 'train':
            args.logger.info("Loading and merging data for modelling ...")
            train = Train(args.variable, logger=args.logger, suffix=args.suffix)
            try:
                train.read_csv_file(vars='Biodiesel_50', fname="{}".format(Config.FILES["BIODIESEL_DF"]))
            except:
                raise
                args.logger.critical("not able to read the product sales data for training ...")
           
            # Perform training 
            args.logger.info("Training the model:")
            if args.district == [None]:
                train.run(algorithms=args.algorithms, model_type=args.model_type)
            fname = "{}_{}{}.{}".format(args.variable, args.model_type, args.suffix, Config.NAME["short"].lower())
            train.save_models(os.path.join(args.output, fname))


        #
        # Perform prediction 
        #       
        elif args.task == 'predict':
            args.logger.info("Loading pre-trained models ...")
            if args.district == [None]:
                args.district = None
            analysis = Analysis(district=args.district, suffix=args.suffix, logger=args.logger)
            train = Train(args.variable, logger=args.logger, suffix=args.suffix)
            try:
                fname = os.path.join(args.input, args.variable + '_' + args.model_type + args.suffix + '.' + Config.NAME["short"].lower())
                train.load_models(fname)
            except:
                raise
                args.logger.critical("not able to read the pre-fit model from file {}.".format(fname))

            try:
                args.logger.info(" Loading Sales Data ...")
                analysis.get_biodiesel()
                analysis.get_primax95()
            except:
                raise
                args.logger.critical("not able to read the new data from files.")

            try:
                train.read_csv_file(vars='Biodiesel_50', fname="{}{}.csv".format(Config.FILES["BIODIESEL_DF"], args.suffix))
            except:
                raise
                args.logger.critical("not able to read the product sales data for training ...")

            df_result =  train.predict(train.data, clustering_by=args.model_type)
            if df_result.size > 0:
                fname = os.path.join(args.input, '{}{}_{}_output.csv'.format(args.variable, args.suffix, args.model_type))
                df_result.to_csv(fname)
                args.logger.info("    prediction results saved to file {}.".format(fname))
                args.logger.info("prediction successfuly done for {} strings.".format(n_wells))
            else:
                args.logger.error("no string found in the loaded models for variable '{}'.".format(args.variable))
    except:
        raise
    

class Parse(Base_Parse):

    def __init__(self):
        self.parser = argparse.ArgumentParser(prog=os.path.split(__current_module__)[1], 
                                            description='Description: '+ __description__, 
                                            epilog='Petronas Confidential - Author: {}'.format(__author__))        
        subparsers = self.parser.add_subparsers(dest='task')
        subparsers.required = True

        report  = subparsers.add_parser('report', help='create HTML reports from Jupyter notebooks ')
        report.add_argument('-i', '--input', dest='input', required=True,
                            help="Jupyter input file containing report.")
        report.add_argument('-x', '--exec', dest='exec', action='store_true', 
                            help="execute the notebook before exporting.")
        report.add_argument('-t', '--template', dest='template', default='basic', 
                            help="template used for report.")
        report.add_argument('-o', '--output', dest='output', default=Config.FILES["DATA_LOCAL"], 
                            help="Output file path and name, default is '../data_local/PWM_<DATE>-<TIME>.csv'.")


        analysis = subparsers.add_parser('analysis', help='performs analysis of features in the input dataset')
        analysis.add_argument("-t", '--data_types', dest='data_types', nargs='+', default=[],
                            help="Download data from database; default data sources are ...")
        analysis.add_argument("-d", '--district', dest='district', nargs='+', default=None,
                            help="""selected district(s) name for which we do the training only. By default all the district are in dataset.""")
        analysis.add_argument("-a", '--state', dest='state', nargs='+', default='',
                            help="""selected state(s) abbreviation in the "State" column. By default they are defined in the Config.py file.""")
        analysis.add_argument('-i', '--input', dest='input', default=Config.FILES["DATA_LOCAL"],
                            help="input directory containing raw data file; default is 'data_local'.")
        analysis.add_argument('-s', '--suffix', dest='suffix', default='', 
                            help="suffix for in the input file name, default is '' means the file name will be named after the table names only.")
							

        train = subparsers.add_parser('train', help='train the model using cleaned data')
        train.add_argument('-i', '--input', dest='input', default=os.path.join(Config.FILES["DATA_LOCAL"], Config.FILES["DATA_LOCAL"]),
                            help="input data file containing cleaned data (output of analysis task); default is 'data_local/SA_CLEANED_DATA.csv'.")
        train.add_argument('-k', '--key', dest='key', default='Biodiesel_50', choices=Config.MODELLING_CONFIG["VAR_KEYS"],
                            help="type of product interested to forecast; default is 'Biodiesel_50'.")
        train.add_argument('-a',  '--algorithms', dest='algorithms', nargs='+', default=["XGBR", "LGBMR", "RFR", "XGBR_tuned", "LGBMR_tuned"],
                            help="algorithm set name, default is 'XGBR', 'LGBMR', 'RFR', 'XGBR_tuned', 'LGBMR_tuned', 'RFR_tuned'.")
        train.add_argument('-m',  '--model_type', dest='model_type', default='Supervised', choices=Config.MODELLING_CONFIG["MODEL_TYPE"],
                            help="model type, default is 'Supervised'.")
        train.add_argument('-u',  '--univariate', dest='univariate', action='store_true',
                            help="choose between either univariate or multivariate forecasting.")
        train.add_argument("-w", '--district', dest='district', nargs='+', default=None,
                            help="""selected district(s) name for which we do the training only. By default all the districts are in dataset.""")
        train.add_argument('-v',  '--variable', dest='variable', default='Prod_Sales', 
                            help="variable to be modelled, default is 'Prod_Sales'.")
        train.add_argument('-o',  '--output', dest='output', default=os.path.join(Config.FILES["DATA"], Config.FILES["MODELS"]),
                            help="output directory name, default is model directory src/data/models")
        train.add_argument('-f',  '--file_name', dest='file_name', default='',
                            help="output file_name, default is empty means <VAR>.sa")
        train.add_argument('-s', '--suffix', dest='suffix', default='', 
                            help="suffix for in the input file name, default is '' means the file name will be named after the table names only.")


        predict = subparsers.add_parser('predict', help='predict using a pre-fit model')
        predict.add_argument('-i', '--input', dest='input', default=os.path.join(Config.FILES["DATA"], Config.FILES["MODELS"]),
                            help="input directory containing pre-fit model '<var>.sa', default is 'data/models' directory.")
        predict.add_argument("-w", '--district', dest='district', nargs='+', default=None,
                            help="""selected district(s) name for which we do the training only. By default all the districts are in dataset.""")
        predict.add_argument('-m',  '--model_type', dest='model_type', default='Supervised', choices=Config.MODELLING_CONFIG["MODEL_TYPE"],
                            help="model type, default is 'Supervised'.")
        predict.add_argument('-s', '--suffix', dest='suffix', default='', 
                            help="suffix for in the input file name, default is '' means the file name will be named after the table names only.")
        predict.add_argument('-v',  '--variable', dest='variable', default='Prod_Sales', 
                            help="variable to be modelled, default is 'Prod_Sales'.")
        predict.add_argument('-o', '--output', dest='output', default=Config.FILES["DATA_LOCAL"],
                            help="Output directory to store result files in.")
                

        self.parser.add_argument("-i", '--idb_name', dest='idb_name', default='Petronas_PDB',
                            help="the input database name or input cvs file name; default is 'Petronas_PDB'.")
        self.parser.add_argument("-n", '--name', dest='table_name', default='',
                            help="Table (view) names in the input database containing the dataset; default is tables defined in 'Config.py'.")
        self.parser.add_argument("-t", '--odb_connstr', dest='odb_connstr', default='', choices=["", "sqlite3"],
                            help="Connection  string for the database containing output table; default is '' means no output to database.")
        self.parser.add_argument("-v", '--version', action='version', version='%(prog)s {0}'.format(__version__) )

