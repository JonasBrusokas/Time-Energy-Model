import numpy as np

from utilz import *


class SFDBLogger:
    def __init__(
        self,
        args: dict,  
        database_file_path: Optional[str] = None,
    ):
        from contrib.SingleFileDB import sfdb

        if database_file_path is None:
            raise ValueError(f"Please provide a database file path!")

        self.database_file_path = database_file_path
        try:
            database_file_parent = Path(database_file_path).parent
            if not database_file_parent.exists():
                print(
                    f"Parent path does not exist: {database_file_parent}. Creating for SFDB!"
                )
                FileUtils.create_dir(database_file_parent)
            self.db = sfdb.Database(database_file_path)
        except Exception as e:
            print(
                f"Failed to initialize the database: {e}. Path='{database_file_path}'"
            )
            raise e
        self.args = args

    def generate_uuid_for_row(self) -> str:
        import uuid

        generated_uuid = uuid.uuid4()
        return str(generated_uuid)

    def _close_db(self):
        try:
            self.db.commit()
            self.db.close()
        except Exception as e:
            print(
                f"Error while closing database (path={self.database_file_path}): \n{e}"
            )

    def close(self):
        self._close_db()

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

    
    
    def _sanitize_value(self, value):
        if isinstance(value, np.float32):
            return np.float64(value)
        else:
            return value

    def report_scalar(self, title, series, value, iteration=0, report_args=False):
        data_object_to_insert = {
            "title": title,
            "series": series,
            "value": self._sanitize_value(value),
            "iteration": iteration,
            "type_of_record": "log",
        }
        if report_args:
            data_object_to_insert["args"] = str(self.args)
        gen_uuid = self.generate_uuid_for_row()
        self.db[gen_uuid] = data_object_to_insert
        self.db.commit()
