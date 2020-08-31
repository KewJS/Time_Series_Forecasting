import sys
import werkzeug
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from flask import Flask
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore')

sys.path.append('src')

# from src import create_app
# sa = create_app()

# flask_app = Flask(__name__)

# @flask_app.route('/')
# def index():
#     return 'Hi, welcome to time series forecasting!'

# application = DispatcherMiddleware(flask_app, {
#      '/sa': sa,
# }) 

""" =====================================================================
  run standalone                                                         |
======================================================================="""
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sa.run('localhost', 8051, debug=True)
    else:
        from src.cmd import main as sa_main
        sa_main(sys.argv[1:])

    #run_simple('localhost', 8050, application, use_reloader=True, use_debugger=True)