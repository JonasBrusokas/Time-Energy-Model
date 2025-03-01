import atexit
import numpy as np

from external_libs.sfdb_logger import SFDBLogger
from utilz import *


class VanillaLogger:
    def __init__(self, logger):
        self.logger = logger

    def report_scalar(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)


class LazyLogger:
    def __init__(self, logger):
        """
        Structure:
        {'title': {'series': {'iteration(int)': 'value(double)'}}}
        """
        self.lazy_logged_value_dict = {}
        self.logger = logger

    def _set_val(self, title: str, series: str, value: float, iteration: int):
        if title not in self.lazy_logged_value_dict:
            
            self.lazy_logged_value_dict[title] = {}

        if series not in self.lazy_logged_value_dict[title]:
            
            self.lazy_logged_value_dict[title][series] = {iteration: value}
        else:
            
            self.lazy_logged_value_dict[title][series][iteration] = value

    def report_scalar(
        self,
        title,
        series,
        value,
        iteration,
    ):
        self._set_val(title, series, value, iteration)

    def report_table(self, title, series, iteration, table_plot):
        self.logger.report_table(
            title=title, series=series, iteration=iteration, table_plot=table_plot
        )

    def close(self):
        for title_key in self.lazy_logged_value_dict.keys():
            for series_key in self.lazy_logged_value_dict[title_key].keys():
                arr = np.array(
                    list(self.lazy_logged_value_dict[title_key][series_key].items())
                )
                self.logger.report_scatter2d(
                    title_key,
                    series_key,
                    iteration=0,
                    scatter=arr,
                    xaxis="Iteration",
                    yaxis="Value",
                    mode="lines+markers",
                )




class FakeLogger:
    def report_table(
        self,
        title,
        series,
        iteration=None,
        table_plot=None,
        csv=None,
        url=None,
        extra_layout=None,
    ):
        print("report_table")

    def report_scalar(self, title, series, value, iteration):
        pass
        




class SFDB_logger:
    def __init__(
        self,
        args: dict,  
        project_name: str = "EBM Research",
        experiment_name: str = "Random Experiment",
        test_mode: bool = False,
        lazy_mode: bool = False,
        path=None,
    ):
        self.test_mode = test_mode

        
        
        
        
        
        
        
        
        
        
        
        

        
        
        

        self.logger = SFDBLogger(args=args, database_file_path=path)

    def task_connect(self, mutable, name=None):
        if not self.test_mode:
            self.task.connect(mutable, name)

    def get_logger(self):
        return self.logger

    def close_clearml(self):
        
        if not self.test_mode:
            print(f"At exit fired!")
            if self.lazy_mode:
                self.logger.close()

            self.task.mark_completed()
            self.task.close()
            self.task = None

    def register_at_exit(self):
        if not self.test_mode:
            atexit.register(self.close_clearml, self.task)





if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    sfdb_logger = SFDBLogger(
        {},
        os.path.join(
            FileUtils.project_root_dir(),
            "temp_sfdb.db",
        ),
    )

    for i in range(10):
        sfdb_logger.report_scalar("some_title", "some_series", {"some_value": i}, i)
