import pandas as pd
from sqlalchemy import create_engine, text
import math
import json
import os

from decimal import Decimal
from datetime import date, datetime
import uuid

def default_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode(errors='replace')  # 或使用 base64 编码
    raise TypeError(f"Type {type(obj)} not serializable")

class SQL_Alchemy_Helper:
    def __init__(self, config):
        # SQLite: file-based, tanpa server.
        db_path = config.get("sqlite_path", "./tablerag.db")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")

    def execute_sql(self, sql, args=None):
        """
        执行 insert/update/delete
        """
        with self.engine.begin() as conn:
            conn.execute(text(sql), args or {})

    def fetchall(self, sql, args=None):
        """
        执行 select 查询
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), args or {})
            rows = result.fetchall()

            json_result = [dict(row._mapping) for row in rows]

            json_result_str = json.dumps(json_result, ensure_ascii=False,  default=default_serializer)

            if len(json_result_str) > 1000:
                return json_result_str[:1000]
            else:
                return json_result_str

    def fetch_dataframe(self, sql, args=None):
        """
        查询结果转为 DataFrame
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=args)
            return df

    def insert_dataframe_batch(self, df, table_name, batch_size=1000):
        """
        DataFrame 批量插入
        """
        df.to_sql(table_name, self.engine, index=False, if_exists='replace', chunksize=batch_size, method='multi')